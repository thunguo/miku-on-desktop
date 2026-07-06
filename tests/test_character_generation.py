"""`miku_on_desk.character_generation` 中确定性后处理纯函数的回归测试。

只测试不需要真实 API 调用的部分（帧切割/透明清理/调色板量化/拼装/QA 健全性检查）；
`generate_reference_image`/`generate_state_strip` 需要真实计费的 OpenAI 调用，按计划
必须由实现者手动跑一次并做人工视觉 QA，不在自动化测试范围内。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from openai.types.image import Image as OpenAIImage
from openai.types.images_response import ImagesResponse
from PIL import Image

from miku_on_desk.character_generation import (
    GenerationConfig,
    GenerationError,
    StatePromptSpec,
    _cleanup_transparency,
    _content_bbox,
    _crop_to_bbox,
    _decode_first_image,
    _effective_background,
    _quantize_preserving_alpha,
    assemble_spritesheet,
    postprocess_strip,
    qa_check,
)
from miku_on_desk.face.pet_state import PetState
from miku_on_desk.face.sprite_sheet import SpriteSheetMeta, StateSpriteInfo, cell_rect


def _solid_frame(color: tuple[int, int, int, int], size: tuple[int, int]) -> Image.Image:
    return Image.new("RGBA", size, color)


def _make_strip(
    colors: list[tuple[int, int, int, int]], frame_size: tuple[int, int]
) -> Image.Image:
    width, height = frame_size
    strip = Image.new("RGBA", (width * len(colors), height))
    for i, color in enumerate(colors):
        strip.paste(_solid_frame(color, frame_size), (i * width, 0))
    return strip


def test_postprocess_strip_splits_into_correct_frame_count_and_size() -> None:
    strip = _make_strip(
        [(255, 0, 0, 0), (0, 255, 0, 0), (0, 0, 255, 0), (255, 255, 0, 0)], (64, 64)
    )

    result = postprocess_strip(strip, frame_count=4, frame_width=32, frame_height=32)

    assert result.size == (32 * 4, 32)


def test_postprocess_strip_with_already_transparent_background_is_untouched() -> None:
    """四角本来就是透明的（alpha < 250）时,_cleanup_transparency 应直接放行,
    不应该误判/误清理已经正确的透明背景。
    """
    strip = _make_strip([(200, 50, 50, 0)], (16, 16))

    result = postprocess_strip(
        strip, frame_count=1, frame_width=16, frame_height=16, palette_colors=8
    )

    array = np.array(result)
    assert array[:, :, 3].max() == 0


def test_cleanup_transparency_removes_uniform_opaque_corner_background() -> None:
    """模拟 API 忽略 transparent 请求、退化返回纯白背景的情形：四角同色不透明,
    中间是另一种颜色的前景方块——应被泛洪填充清理成透明背景,前景保留不透明。
    """
    size = 40
    array = np.full((size, size, 4), (255, 255, 255, 255), dtype=np.uint8)
    array[10:30, 10:30] = (10, 20, 200, 255)
    frame = Image.fromarray(array, mode="RGBA")

    cleaned = _cleanup_transparency(frame)
    cleaned_array = np.array(cleaned)

    assert cleaned_array[0, 0, 3] == 0
    assert cleaned_array[0, size - 1, 3] == 0
    assert cleaned_array[20, 20, 3] == 255


def test_cleanup_transparency_leaves_non_uniform_corners_untouched() -> None:
    """四角颜色不一致时不是单色背景,不应盲目清理,否则可能误删前景细节。"""
    size = 20
    array = np.zeros((size, size, 4), dtype=np.uint8)
    array[:, :, 3] = 255
    array[0, 0] = (255, 0, 0, 255)
    array[0, size - 1] = (0, 255, 0, 255)
    array[size - 1, 0] = (0, 0, 255, 255)
    array[size - 1, size - 1] = (255, 255, 0, 255)
    frame = Image.fromarray(array, mode="RGBA")

    cleaned = _cleanup_transparency(frame)

    assert np.array_equal(np.array(cleaned), array)


def test_quantize_preserving_alpha_keeps_alpha_channel_exact() -> None:
    size = 10
    array = np.zeros((size, size, 4), dtype=np.uint8)
    array[:, : size // 2, :3] = (10, 20, 30)
    array[:, size // 2 :, :3] = (200, 210, 220)
    array[:, :, 3] = np.linspace(0, 255, size, dtype=np.uint8)[:, None]
    frame = Image.fromarray(array, mode="RGBA")

    quantized = _quantize_preserving_alpha(frame, colors=4)

    quantized_alpha = np.array(quantized)[:, :, 3]
    assert np.array_equal(quantized_alpha, array[:, :, 3])


def test_content_bbox_returns_union_of_alpha_across_frames() -> None:
    frame_a = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    frame_a.paste(Image.new("RGBA", (4, 4), (255, 0, 0, 255)), (5, 5))
    frame_b = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    frame_b.paste(Image.new("RGBA", (10, 10), (0, 0, 255, 255)), (20, 20))

    bbox = _content_bbox([frame_a, frame_b])

    assert bbox == (5, 5, 30, 30)


def test_content_bbox_returns_none_when_all_frames_fully_transparent() -> None:
    frame = Image.new("RGBA", (10, 10), (0, 0, 0, 0))

    assert _content_bbox([frame, frame]) is None


def test_crop_to_bbox_adds_padding_and_clamps_to_frame_bounds() -> None:
    frame = Image.new("RGBA", (20, 20), (0, 0, 0, 0))

    cropped = _crop_to_bbox(frame, (5, 5, 15, 15), padding_ratio=0.5)

    assert cropped.size == (20, 20)


def test_postprocess_strip_scales_all_frames_by_shared_content_bbox() -> None:
    """两帧内容尺寸差异很大时（一帧内容很小,一帧内容几乎占满源画布）,应使用同一套
    裁剪窗口统一缩放,而不是让内容更大的帧被相对缩小——否则同一动作序列里角色大小会
    逐帧跳变,比整体偏小但缩放一致更影响观感。
    """
    size = 40
    frame_a = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    frame_a.paste(Image.new("RGBA", (4, 4), (255, 0, 0, 255)), (18, 18))
    frame_b = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    frame_b.paste(Image.new("RGBA", (32, 32), (0, 0, 255, 255)), (4, 4))
    strip = Image.new("RGBA", (size * 2, size), (0, 0, 0, 0))
    strip.paste(frame_a, (0, 0))
    strip.paste(frame_b, (size, 0))

    result = postprocess_strip(
        strip, frame_count=2, frame_width=size, frame_height=size, palette_colors=8
    )

    out_a = np.array(result.crop((0, 0, size, size)))
    out_b = np.array(result.crop((size, 0, size * 2, size)))
    width_a = np.count_nonzero(out_a[:, :, 3].any(axis=0))
    width_b = np.count_nonzero(out_b[:, :, 3].any(axis=0))
    assert width_b > width_a * 3


def test_assemble_spritesheet_places_rows_in_spec_order_and_builds_matching_meta() -> None:
    frame_width, frame_height = 8, 8
    config = GenerationConfig(
        pet_name="test_pet",
        description="unused",
        output_dir=Path("/tmp/unused"),
        frame_width=frame_width,
        frame_height=frame_height,
    )
    specs = (
        StatePromptSpec(state=PetState.IDLE, frame_count=2, fps=5.0, loop=True, pose_prompt="idle"),
        StatePromptSpec(
            state=PetState.SUCCESS, frame_count=1, fps=8.0, loop=False, pose_prompt="success"
        ),
    )
    strips = [
        _make_strip([(255, 0, 0, 255), (0, 255, 0, 255)], (frame_width, frame_height)),
        _make_strip([(0, 0, 255, 255)], (frame_width, frame_height)),
    ]

    sheet, meta = assemble_spritesheet(strips, specs, config)

    assert sheet.size == (frame_width * 2, frame_height * 2)
    assert meta.states[PetState.IDLE].row == 0
    assert meta.states[PetState.SUCCESS].row == 1
    assert meta.states[PetState.SUCCESS].frame_count == 1
    assert meta.states[PetState.SUCCESS].loop is False

    success_rect = cell_rect(meta, PetState.SUCCESS, 0)
    pixel = sheet.getpixel((success_rect.x, success_rect.y))
    assert pixel == (0, 0, 255, 255)


def _meta_with_single_state(
    *, frame_count: int, frame_width: int, frame_height: int
) -> SpriteSheetMeta:
    return SpriteSheetMeta(
        pet_name="test_pet",
        frame_width=frame_width,
        frame_height=frame_height,
        columns=frame_count,
        rows=1,
        fallback_state=PetState.IDLE,
        states={PetState.IDLE: StateSpriteInfo(row=0, frame_count=frame_count, fps=5.0, loop=True)},
    )


def test_qa_check_passes_for_fully_painted_sheet() -> None:
    frame_width, frame_height, frame_count = 4, 4, 2
    meta = _meta_with_single_state(
        frame_count=frame_count, frame_width=frame_width, frame_height=frame_height
    )
    sheet = Image.new("RGBA", (frame_width * frame_count, frame_height), (10, 20, 30, 255))

    assert qa_check(sheet, meta) == []


def test_qa_check_flags_fully_transparent_frame() -> None:
    frame_width, frame_height, frame_count = 4, 4, 2
    meta = _meta_with_single_state(
        frame_count=frame_count, frame_width=frame_width, frame_height=frame_height
    )
    sheet = Image.new("RGBA", (frame_width * frame_count, frame_height), (0, 0, 0, 0))
    sheet.paste(Image.new("RGBA", (frame_width, frame_height), (10, 20, 30, 255)), (0, 0))

    problems = qa_check(sheet, meta)

    assert len(problems) == 1
    assert "第 1 帧" in problems[0]


def test_qa_check_flags_sheet_size_mismatch() -> None:
    meta = _meta_with_single_state(frame_count=2, frame_width=4, frame_height=4)
    wrong_size_sheet = Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    problems = qa_check(wrong_size_sheet, meta)

    assert len(problems) == 1
    assert "尺寸" in problems[0]


def test_decode_first_image_raises_clear_error_for_non_json_string_response() -> None:
    """openai SDK 在自定义 base_url 返回非 JSON Content-Type 时会静默把响应体退化成
    str（而不是抛异常）,这里必须提前拦截,而不是让调用方摸到一句无从下手的
    'str' object has no attribute 'data'。
    """
    with pytest.raises(GenerationError, match="非预期的内容"):
        _decode_first_image("<html>502 Bad Gateway</html>")


def test_decode_first_image_raises_when_data_missing_b64_json() -> None:
    response = ImagesResponse.model_construct(data=[OpenAIImage.model_construct(b64_json=None)])

    with pytest.raises(GenerationError, match="未返回 b64_json"):
        _decode_first_image(response)


def _config_with(
    model: str, background: Literal["transparent", "opaque", "auto"]
) -> GenerationConfig:
    return GenerationConfig(
        pet_name="test_pet",
        description="unused",
        output_dir=Path("/tmp/unused"),
        model=model,
        background=background,
    )


def test_effective_background_downgrades_transparent_to_opaque_for_gpt_image_2() -> None:
    """gpt-image-2 系列型号请求 transparent 背景会被 API 直接 400,必须提前降级成
    opaque,再靠 postprocess_strip 里已有的 _cleanup_transparency 泛洪填充兜底。
    """
    config = _config_with("gpt-image-2", "transparent")

    assert _effective_background(config) == "opaque"


def test_effective_background_leaves_transparent_untouched_for_gpt_image_1() -> None:
    config = _config_with("gpt-image-1", "transparent")

    assert _effective_background(config) == "transparent"


def test_effective_background_does_not_override_explicit_non_transparent_choice() -> None:
    config = _config_with("gpt-image-2", "auto")

    assert _effective_background(config) == "auto"
