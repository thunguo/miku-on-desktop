"""角色生成流水线：先锁定一张基准参考图，再为每个 `PetState` 条件生成一条横向多帧动作
序列，最后确定性拼装成完整 spritesheet + `pet.json`。

供 ``scripts/generate_pet_sprites.py``（CLI，需要真实计费，极少运行）与
``face/character_generation_worker.py``（GUI 角色创建对话框的后台线程）共享同一条实现——
``scripts/`` 目录故意不是本包的一部分（不注册为 console-script），GUI 运行时需要的是
一次正常的包内 import，因此把可复用逻辑放在这里，而不是让 GUI 反过来动态加载
``scripts/`` 下的文件。

设计角色而非复刻任何现有模型：本模块生成的是 AI 原创像素画，不以
`assets/legacy_pmx/` 下任何模型的渲染截图作为生成参考图（详见
`assets/legacy_pmx/README.md` 中的授权说明）。
"""

from __future__ import annotations

import base64
import io
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from openai import OpenAI
from openai.types.images_response import ImagesResponse
from PIL import Image

from miku_on_desk.face.pet_state import PetState
from miku_on_desk.face.sprite_sheet import SpriteSheetMeta, StateSpriteInfo, cell_rect

# 1536 能被本模块用到的所有 frame_count（2/4/6/8）整除，保证等分帧不产生舍入误差。
_STRIP_SIZE = "1536x1024"
_BACKGROUND_COLOR_TOLERANCE = 24

_REFERENCE_PROMPT_INTRO_TEXT_ONLY = """\
Design a single original chibi pixel-art character reference sheet for a desktop pet.
Character concept: {description}
"""

_REFERENCE_PROMPT_INTRO_WITH_IMAGE = """\
Using the attached image purely as visual/thematic inspiration (do not copy it verbatim;
reinterpret its colors, silhouette, and defining features), design a single original chibi
pixel-art character reference sheet for a desktop pet that combines the attached image's
visual identity with this text concept: {description}
"""

_REFERENCE_PROMPT_INTRO_SELFIE = """\
The attached photo is a real selfie/webcam capture of a real person, and the framing likely only
shows their head, face, and shoulders (no legs or feet visible). Use it purely as visual/thematic
inspiration for an ORIGINAL chibi mascot — reinterpret the person's most distinctive visible traits
(hair color/style/length, skin tone, glasses, notable clothing colors, general vibe) rather than
reproducing the photo. You must INVENT a complete, plausible full body that is not shown in the
photo — torso, arms, legs, and feet — with a simple, cute outfit consistent with what little
clothing is visible near the shoulders. Do NOT reproduce the photo's cropping, camera angle, or
composition; this is a full-body character design task, and the input photo only supplies
facial/hair identity, nothing about the lower body. Design a single original chibi pixel-art
character reference sheet combining this visual identity with this concept: {description}
"""

_REFERENCE_PROMPT_STYLE_BODY = """\
Art style requirements (must look like genuine retro low-resolution pixel art, not a smooth
digital illustration): render as if drawn on a small ~48x48 pixel grid and then scaled up with
hard nearest-neighbor edges — large, clearly visible square pixel blocks, flat cel-shaded colors
with a limited palette (roughly 16-32 colors), thick clean dark outlines, hard edges only,
absolutely no anti-aliasing, no smooth gradients, no soft shading, no fine painterly detail.
Think classic 16-bit era game sprite (SNES/Game Boy Advance), not a painting.
Composition: front-facing, full body with head and feet both clearly visible, extra-cute
super-deformed proportions (oversized head, roughly 2 heads tall total), simplified expressive
face, transparent background, centered composition with even padding, neutral standing idle
pose, consistent silhouette suitable for being reused as a style reference for further
animation frames. Do not include any text, watermark, or logo in the image.
"""

_STRIP_PROMPT_TEMPLATE = """\
Using the attached reference image as the definitive character design (same colors,
proportions, outfit, and chunky low-resolution pixel-art style), draw a horizontal filmstrip
of exactly {frame_count} equal-width animation frames arranged left to right, depicting:
{pose_prompt}
Keep rendering as genuine large-pixel retro sprite art: hard nearest-neighbor edges, flat
cel-shaded limited-palette colors, thick clean outlines, no anti-aliasing, no smooth
gradients, no fine painterly detail — consistent with the reference image's blocky pixel
grid, not a smooth illustration. Keep the character's design, palette, and pixel-art style
perfectly consistent with the reference image across all frames, with full body (head to
feet) clearly visible in every frame. Transparent background. No text, no panel borders, no
frame numbering, no watermark. Each frame must be tightly and identically framed (same
camera distance and crop) so the frames can be split into equal-width tiles.
"""


class GenerationError(Exception):
    """图像生成 API 响应缺少可用的图像数据。"""


# gpt-image-2 系列型号明确不支持透明背景，请求 background="transparent" 会被 API
# 直接 400（而不是像其他型号那样静默退化成不透明背景）——这份名单抄自 openai SDK
# 自己 images.edit 的文档说明，不是我们猜的。
_TRANSPARENT_BACKGROUND_UNSUPPORTED_MODELS = frozenset({"gpt-image-2", "gpt-image-2-2026-04-21"})


def _effective_background(config: GenerationConfig) -> Literal["transparent", "opaque", "auto"]:
    if (
        config.background == "transparent"
        and config.model in _TRANSPARENT_BACKGROUND_UNSUPPORTED_MODELS
    ):
        return "opaque"
    return config.background


@dataclass(frozen=True)
class GenerationConfig:
    pet_name: str
    description: str
    output_dir: Path
    frame_width: int = 128
    frame_height: int = 128
    model: str = "gpt-image-1"
    api_key: str | None = None
    base_url: str | None = None
    reference_image_path: Path | None = None
    reference_image_kind: Literal["illustration", "selfie"] = "illustration"
    background: Literal["transparent", "opaque", "auto"] = "transparent"


@dataclass(frozen=True)
class StatePromptSpec:
    state: PetState
    frame_count: int
    fps: float
    loop: bool
    pose_prompt: str


# 单一数据源：同时驱动 pet.json 里的逐状态参数与生成时的逐状态 prompt。行号即
# tuple 下标，与 face/sprite_sheet.py 的 pet.json 行布局约定一致。
STATE_SPECS: tuple[StatePromptSpec, ...] = (
    StatePromptSpec(
        state=PetState.IDLE,
        frame_count=6,
        fps=6.0,
        loop=True,
        pose_prompt=(
            "a calm idle breathing loop: gentle up-and-down bobbing with slow blinking, "
            "subtle hair sway, looping seamlessly from the last frame back to the first"
        ),
    ),
    StatePromptSpec(
        state=PetState.TALKING,
        frame_count=8,
        fps=10.0,
        loop=True,
        pose_prompt=(
            "a talking/mouth-flapping loop: mouth opening and closing in various shapes as "
            "if speaking energetically, slight head tilt, looping seamlessly"
        ),
    ),
    StatePromptSpec(
        state=PetState.THINKING,
        frame_count=6,
        fps=5.0,
        loop=True,
        pose_prompt=(
            "a thinking loop: head tilted, one hand near the chin, eyes looking upward as "
            "if pondering, slow contemplative motion, looping seamlessly"
        ),
    ),
    StatePromptSpec(
        state=PetState.TOOL_RUNNING,
        frame_count=8,
        fps=8.0,
        loop=True,
        pose_prompt=(
            "a busy-working loop: character actively typing on a floating keyboard with a "
            "focused expression, quick repetitive arm motion, looping seamlessly"
        ),
    ),
    StatePromptSpec(
        state=PetState.CONFIRMATION_PENDING,
        frame_count=4,
        fps=3.0,
        loop=True,
        pose_prompt=(
            "a questioning loop: head tilted with a curious raised eyebrow and a small "
            "question-mark expression, gentle swaying, looping seamlessly"
        ),
    ),
    StatePromptSpec(
        state=PetState.DRAGGED,
        frame_count=2,
        fps=4.0,
        loop=True,
        pose_prompt=(
            "a being-picked-up loop: surprised wide-eyed expression with arms and legs "
            "slightly dangling as if lifted by the scruff, a two-frame back-and-forth wobble"
        ),
    ),
    StatePromptSpec(
        state=PetState.SUCCESS,
        frame_count=6,
        fps=10.0,
        loop=False,
        pose_prompt=(
            "a one-time cheerful success celebration: jumping up with both arms raised and "
            "a big happy smile, ending in a settled joyful pose, sequential (not looping) "
            "from start to finish"
        ),
    ),
    StatePromptSpec(
        state=PetState.ERROR,
        frame_count=6,
        fps=10.0,
        loop=False,
        pose_prompt=(
            "a one-time distressed error reaction: startled flinch with a worried "
            "expression and a small sweat drop, ending in a slightly slumped pose, "
            "sequential (not looping) from start to finish"
        ),
    ),
    StatePromptSpec(
        state=PetState.CLICKED,
        frame_count=4,
        fps=12.0,
        loop=False,
        pose_prompt=(
            "a one-time poked reaction: a brief surprised squish/bounce as if poked, "
            "ending back at a neutral standing pose, sequential (not looping) from start "
            "to finish"
        ),
    ),
    StatePromptSpec(
        state=PetState.NOTICE,
        frame_count=4,
        fps=8.0,
        loop=False,
        pose_prompt=(
            "a one-time attention-getting notice reaction: perking up with a small alert "
            "sparkle near the head and an attentive expression, ending in an attentive "
            "standing pose, sequential (not looping) from start to finish"
        ),
    ),
)


def _decode_first_image(response: ImagesResponse | str) -> Image.Image:
    if isinstance(response, str):
        # 自定义 base_url（第三方代理）返回非 JSON Content-Type 时,openai SDK 不会报错,
        # 而是静默把原始响应文本当字符串返回——此时 response.data 会在调用方炸成一句
        # 无从下手的 "'str' object has no attribute 'data'"。这里提前拦截,把代理实际
        # 返回的内容原样带出来,方便定位是鉴权失败、模型不支持还是网关错误页。
        raise GenerationError(
            f"API 返回了非预期的内容（可能是 base_url 配置有误）：{response[:500]}"
        )
    if not response.data or response.data[0].b64_json is None:
        raise GenerationError("API 未返回 b64_json 图像数据")
    raw = base64.b64decode(response.data[0].b64_json)
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def generate_reference_image(client: OpenAI, config: GenerationConfig) -> Image.Image:
    if config.reference_image_path is not None:
        return _generate_reference_image_from_upload(client, config)
    prompt = _REFERENCE_PROMPT_INTRO_TEXT_ONLY.format(
        description=config.description
    ) + _REFERENCE_PROMPT_STYLE_BODY
    response = client.images.generate(
        model=config.model,
        prompt=prompt,
        size="1024x1024",
        quality="high",
        background=_effective_background(config),
        output_format="png",
    )
    return _decode_first_image(response)


def _mime_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    return "image/png"


def _reference_image_prompt_for_upload(config: GenerationConfig) -> str:
    intro_template = (
        _REFERENCE_PROMPT_INTRO_SELFIE
        if config.reference_image_kind == "selfie"
        else _REFERENCE_PROMPT_INTRO_WITH_IMAGE
    )
    return intro_template.format(description=config.description) + _REFERENCE_PROMPT_STYLE_BODY


def _generate_reference_image_from_upload(client: OpenAI, config: GenerationConfig) -> Image.Image:
    assert config.reference_image_path is not None
    image_bytes = config.reference_image_path.read_bytes()
    prompt = _reference_image_prompt_for_upload(config)
    response = client.images.edit(
        model=config.model,
        image=(
            config.reference_image_path.name,
            image_bytes,
            _mime_type_for(config.reference_image_path),
        ),
        prompt=prompt,
        size="1024x1024",
        quality="high",
        background=_effective_background(config),
        output_format="png",
    )
    return _decode_first_image(response)


def generate_state_strip(
    client: OpenAI, config: GenerationConfig, reference: Image.Image, spec: StatePromptSpec
) -> Image.Image:
    buffer = io.BytesIO()
    reference.save(buffer, format="PNG")
    prompt = _STRIP_PROMPT_TEMPLATE.format(
        frame_count=spec.frame_count, pose_prompt=spec.pose_prompt
    )
    response = client.images.edit(
        model=config.model,
        image=("reference.png", buffer.getvalue(), "image/png"),
        prompt=prompt,
        size=_STRIP_SIZE,
        quality="high",
        background=_effective_background(config),
        output_format="png",
    )
    return _decode_first_image(response)


def _flood_fill_from_border(mask: np.ndarray) -> np.ndarray:
    """标出所有从图像边界出发、经由 `mask` 中为真的像素四邻域可达的连通区域。

    只从边界出发（而非任意匹配像素），避免把角色前景里恰好同色的区域也当成背景。
    """
    height, width = mask.shape
    visited = np.zeros_like(mask)
    queue: deque[tuple[int, int]] = deque()

    def _seed(y: int, x: int) -> None:
        if mask[y, x] and not visited[y, x]:
            visited[y, x] = True
            queue.append((y, x))

    for x in range(width):
        _seed(0, x)
        _seed(height - 1, x)
    for y in range(height):
        _seed(y, 0)
        _seed(y, width - 1)

    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))
    return visited


def _cleanup_transparency(frame: Image.Image) -> Image.Image:
    """优先信任 API 原生返回的透明背景；仅当四角都已经是不透明的同一种颜色时，
    才假定该颜色是残留背景色并做泛洪填充清理——应对 API 偶尔忽略 transparent 请求、
    退化返回纯色/棋盘背景的情况。四角颜色不一致则放弃清理，避免误删前景。
    """
    array = np.array(frame.convert("RGBA"), dtype=np.uint8)
    height, width = array.shape[:2]
    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    if any(array[y, x, 3] < 250 for y, x in corners):
        return frame

    background = array[0, 0, :3].astype(np.int16)
    corner_diffs = [np.abs(array[y, x, :3].astype(np.int16) - background).max() for y, x in corners]
    if max(corner_diffs) > _BACKGROUND_COLOR_TOLERANCE:
        return frame

    color_distance = np.abs(array[:, :, :3].astype(np.int16) - background).max(axis=2)
    background_mask = color_distance <= _BACKGROUND_COLOR_TOLERANCE
    connected = _flood_fill_from_border(background_mask)
    array[connected, 3] = 0
    return Image.fromarray(array, mode="RGBA")


def _quantize_preserving_alpha(frame: Image.Image, colors: int) -> Image.Image:
    """量化只作用于 RGB 通道，再把（可能已被透明清理过的）alpha 通道原样接回——
    Pillow 的调色板量化不能直接感知 alpha，混着做会把半透明边缘的颜色搅乱。
    """
    rgba = frame.convert("RGBA")
    alpha = rgba.getchannel("A")
    quantized = (
        rgba.convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT).convert("RGBA")
    )
    quantized.putalpha(alpha)
    return quantized


def _resize_contain(frame: Image.Image, frame_width: int, frame_height: int) -> Image.Image:
    """按等比缩放贴进 (frame_width, frame_height) 画布，居中、四周透明留白——

    strip 的每帧切片天然是又窄又高的长条（横向多帧分割 1536×1024 的结果），
    与目标方形帧尺寸的宽高比相差很大；直接 `resize` 会把角色纵向压扁/横向拉宽，
    必须先保持宽高比缩放，再居中贴到目标画布上，避免变形。
    """
    scale = min(frame_width / frame.width, frame_height / frame.height)
    scaled_size = (max(1, round(frame.width * scale)), max(1, round(frame.height * scale)))
    scaled = frame.resize(scaled_size, Image.Resampling.NEAREST)
    canvas = Image.new("RGBA", (frame_width, frame_height), (0, 0, 0, 0))
    offset = ((frame_width - scaled_size[0]) // 2, (frame_height - scaled_size[1]) // 2)
    canvas.paste(scaled, offset, scaled)
    return canvas


def _content_bbox(frames: Sequence[Image.Image]) -> tuple[int, int, int, int] | None:
    """取多帧（局部坐标系一致，因为 prompt 要求同一动作序列所有帧用相同镜头/裁剪）中
    非透明像素的并集外接矩形。

    用并集而不是逐帧各自的 bbox：否则动作幅度较大的帧（例如跳跃时手脚伸展）会因为
    自身 bbox 更大而在 `_resize_contain` 里被相对缩小，造成同一动画里角色大小逐帧
    忽大忽小的抖动——比"整体偏小但所有帧缩放一致"更影响观感。全部帧都全透明（生成
    失败）时返回 None，交给上层 `qa_check` 检测。
    """
    left = top = right = bottom = None
    for frame in frames:
        bbox = frame.getchannel("A").getbbox()
        if bbox is None:
            continue
        frame_left, frame_top, frame_right, frame_bottom = bbox
        left = frame_left if left is None else min(left, frame_left)
        top = frame_top if top is None else min(top, frame_top)
        right = frame_right if right is None else max(right, frame_right)
        bottom = frame_bottom if bottom is None else max(bottom, frame_bottom)
    if left is None or top is None or right is None or bottom is None:
        return None
    return left, top, right, bottom


def _crop_to_bbox(
    frame: Image.Image, bbox: tuple[int, int, int, int], *, padding_ratio: float
) -> Image.Image:
    """裁剪到给定外接矩形，四周留一点比例的透明边距，再交给 `_resize_contain` 缩放——
    去掉源切片里角色周围的大片透明留白，让角色在最终画布里占比更大、更清晰可见。
    """
    left, top, right, bottom = bbox
    width, height = right - left, bottom - top
    pad_x = max(1, round(width * padding_ratio))
    pad_y = max(1, round(height * padding_ratio))
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(frame.width, right + pad_x)
    bottom = min(frame.height, bottom + pad_y)
    return frame.crop((left, top, right, bottom))


def postprocess_strip(
    strip: Image.Image,
    *,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    palette_colors: int = 48,
    content_padding_ratio: float = 0.06,
) -> Image.Image:
    strip = strip.convert("RGBA")
    slice_width = strip.width // frame_count
    frames = [
        _cleanup_transparency(strip.crop((i * slice_width, 0, (i + 1) * slice_width, strip.height)))
        for i in range(frame_count)
    ]
    bbox = _content_bbox(frames)

    row = Image.new("RGBA", (frame_width * frame_count, frame_height), (0, 0, 0, 0))
    for i, frame in enumerate(frames):
        if bbox is not None:
            frame = _crop_to_bbox(frame, bbox, padding_ratio=content_padding_ratio)
        frame = _quantize_preserving_alpha(frame, palette_colors)
        frame = _resize_contain(frame, frame_width, frame_height)
        row.paste(frame, (i * frame_width, 0), frame)
    return row


def assemble_spritesheet(
    strips: Sequence[Image.Image], specs: Sequence[StatePromptSpec], config: GenerationConfig
) -> tuple[Image.Image, SpriteSheetMeta]:
    columns = max(spec.frame_count for spec in specs)
    rows = len(specs)
    sheet = Image.new(
        "RGBA", (columns * config.frame_width, rows * config.frame_height), (0, 0, 0, 0)
    )

    states: dict[PetState, StateSpriteInfo] = {}
    for row_index, (strip, spec) in enumerate(zip(strips, specs, strict=True)):
        sheet.paste(strip, (0, row_index * config.frame_height), strip)
        states[spec.state] = StateSpriteInfo(
            row=row_index, frame_count=spec.frame_count, fps=spec.fps, loop=spec.loop
        )

    meta = SpriteSheetMeta(
        pet_name=config.pet_name,
        frame_width=config.frame_width,
        frame_height=config.frame_height,
        columns=columns,
        rows=rows,
        fallback_state=PetState.IDLE,
        states=states,
    )
    return sheet, meta


def qa_check(sheet: Image.Image, meta: SpriteSheetMeta) -> list[str]:
    """只是廉价的健全性检查（尺寸是否吻合、每格是否非全透明），不能替代人工审美判断。"""
    problems: list[str] = []
    expected_size = (meta.columns * meta.frame_width, meta.rows * meta.frame_height)
    if sheet.size != expected_size:
        problems.append(f"整图尺寸 {sheet.size} 与预期 {expected_size} 不符")
        return problems

    array = np.array(sheet.convert("RGBA"), dtype=np.uint8)
    for state, info in meta.states.items():
        for frame in range(info.frame_count):
            rect = cell_rect(meta, state, frame)
            cell_alpha = array[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width, 3]
            if not cell_alpha.any():
                problems.append(f"状态 {state.value} 第 {frame} 帧完全透明（疑似生成失败）")
    return problems


GenerationStage = Literal["reference", "strip", "assemble", "qa"]


@dataclass(frozen=True)
class GenerationProgress:
    """一次生成流水线里某个阶段的进度快照，供 CLI 打印/GUI 驱动等待动效使用。"""

    stage: GenerationStage
    detail: str
    completed_states: int
    total_states: int
    strip_image: Image.Image | None = None
    reference_image: Image.Image | None = None


class GenerationCancelled(Exception):
    """用户在生成过程中取消。"""


def generate_character(
    config: GenerationConfig,
    *,
    on_progress: Callable[[GenerationProgress], None] = lambda _progress: None,
    should_cancel: Callable[[], bool] = lambda: False,
) -> tuple[Image.Image, SpriteSheetMeta, list[str]]:
    """CLI 与 GUI 共享的完整生成流水线：参考图 → 逐状态动作条 → 拼装 → QA。"""
    client = OpenAI(api_key=config.api_key, base_url=config.base_url or None)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    on_progress(GenerationProgress("reference", "", 0, len(STATE_SPECS)))
    reference = generate_reference_image(client, config)
    reference.save(config.output_dir / "reference.png")
    on_progress(GenerationProgress("reference", "", 0, len(STATE_SPECS), reference_image=reference))
    if should_cancel():
        raise GenerationCancelled()

    raw_dir = config.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    strips = []
    for index, spec in enumerate(STATE_SPECS):
        if should_cancel():
            raise GenerationCancelled()
        raw_strip = generate_state_strip(client, config, reference, spec)
        raw_strip.save(raw_dir / f"{spec.state.value}.png")
        processed = postprocess_strip(
            raw_strip,
            frame_count=spec.frame_count,
            frame_width=config.frame_width,
            frame_height=config.frame_height,
        )
        strips.append(processed)
        on_progress(
            GenerationProgress(
                "strip", spec.state.value, index + 1, len(STATE_SPECS), strip_image=processed
            )
        )

    on_progress(GenerationProgress("assemble", "", len(STATE_SPECS), len(STATE_SPECS)))
    sheet, meta = assemble_spritesheet(strips, STATE_SPECS, config)

    on_progress(GenerationProgress("qa", "", len(STATE_SPECS), len(STATE_SPECS)))
    problems = qa_check(sheet, meta)

    sheet.save(config.output_dir / "spritesheet.png")
    (config.output_dir / "pet.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    return sheet, meta, problems
