"""PetSpriteWidget 的回归测试：验证按 (state, frame) 裁剪出正确的格子、切帧逻辑，以及
放大缩放确实用的是最近邻（而非平滑插值，否则会把像素风格重新糊成渐变）。

用测试内合成的、每个格子颜色可辨识的 spritesheet 夹具，不依赖真实生成的美术资产。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from miku_on_desk.face.pet_state import PetState
from miku_on_desk.face.sprite_sheet import SpriteSheetMeta, StateSpriteInfo
from miku_on_desk.face.ui.sprite_widget import PetSpriteWidget

_FRAME_SIZE = 4
_RED = (255, 0, 0, 255)
_BLUE = (0, 0, 255, 255)
_GREEN = (0, 255, 0, 255)
_MAGENTA = (255, 0, 255, 255)


def _split_frame() -> Image.Image:
    """左半红、右半蓝：用于检测缩放是否引入了原色以外的插值颜色。"""
    frame = Image.new("RGBA", (_FRAME_SIZE, _FRAME_SIZE), _RED)
    frame.paste(Image.new("RGBA", (_FRAME_SIZE // 2, _FRAME_SIZE), _BLUE), (_FRAME_SIZE // 2, 0))
    return frame


def _make_sheet(path: Path) -> SpriteSheetMeta:
    sheet = Image.new("RGBA", (_FRAME_SIZE * 2, _FRAME_SIZE * 2), (0, 0, 0, 0))
    sheet.paste(_split_frame(), (0, 0))
    sheet.paste(Image.new("RGBA", (_FRAME_SIZE, _FRAME_SIZE), _GREEN), (_FRAME_SIZE, 0))
    sheet.paste(Image.new("RGBA", (_FRAME_SIZE, _FRAME_SIZE), _MAGENTA), (0, _FRAME_SIZE))
    sheet.save(path)

    return SpriteSheetMeta(
        pet_name="test_pet",
        frame_width=_FRAME_SIZE,
        frame_height=_FRAME_SIZE,
        columns=2,
        rows=2,
        fallback_state=PetState.IDLE,
        states={
            PetState.IDLE: StateSpriteInfo(row=0, frame_count=2, fps=1.0, loop=True),
            PetState.TALKING: StateSpriteInfo(row=1, frame_count=1, fps=1.0, loop=True),
        },
    )


def _pixel_colors(
    widget: PetSpriteWidget, key: tuple[PetState, int]
) -> set[tuple[int, int, int, int]]:
    image = widget._frames[key].toImage()
    colors = set()
    for y in range(image.height()):
        for x in range(image.width()):
            color: QColor = image.pixelColor(x, y)
            colors.add((color.red(), color.green(), color.blue(), color.alpha()))
    return colors


def test_init_caches_correct_cell_content_for_each_state_and_frame(
    qapp: QApplication, tmp_path: Path
) -> None:
    sheet_path = tmp_path / "spritesheet.png"
    meta = _make_sheet(sheet_path)

    widget = PetSpriteWidget(meta, sheet_path)

    assert _pixel_colors(widget, (PetState.IDLE, 1)) == {_GREEN}
    assert _pixel_colors(widget, (PetState.TALKING, 0)) == {_MAGENTA}


def test_scaling_uses_nearest_neighbor_and_never_blends_colors(
    qapp: QApplication, tmp_path: Path
) -> None:
    """非整数倍缩放下，平滑插值会在红蓝交界处产生第三种混合色；最近邻不会。"""
    sheet_path = tmp_path / "spritesheet.png"
    meta = _make_sheet(sheet_path)

    widget = PetSpriteWidget(meta, sheet_path, scale=1.75)

    colors = _pixel_colors(widget, (PetState.IDLE, 0))
    assert colors <= {_RED, _BLUE}
    assert colors == {_RED, _BLUE}


def test_set_frame_switches_current_pixmap(qapp: QApplication, tmp_path: Path) -> None:
    sheet_path = tmp_path / "spritesheet.png"
    meta = _make_sheet(sheet_path)
    widget = PetSpriteWidget(meta, sheet_path)

    widget.set_frame(PetState.TALKING, 0)

    assert widget._current is widget._frames[(PetState.TALKING, 0)]


def test_set_frame_with_same_key_does_not_call_update(qapp: QApplication, tmp_path: Path) -> None:
    sheet_path = tmp_path / "spritesheet.png"
    meta = _make_sheet(sheet_path)
    widget = PetSpriteWidget(meta, sheet_path)
    widget.set_frame(PetState.IDLE, 1)
    calls = []
    widget.update = lambda: calls.append(1)  # type: ignore[method-assign]

    widget.set_frame(PetState.IDLE, 1)

    assert calls == []


def test_set_frame_calls_update_when_frame_actually_changes(
    qapp: QApplication, tmp_path: Path
) -> None:
    sheet_path = tmp_path / "spritesheet.png"
    meta = _make_sheet(sheet_path)
    widget = PetSpriteWidget(meta, sheet_path)
    calls = []
    widget.update = lambda: calls.append(1)  # type: ignore[method-assign]

    widget.set_frame(PetState.IDLE, 1)

    assert calls == [1]


def test_state_missing_from_asset_falls_back_to_fallback_state_row(
    qapp: QApplication, tmp_path: Path
) -> None:
    sheet_path = tmp_path / "spritesheet.png"
    meta = _make_sheet(sheet_path)
    widget = PetSpriteWidget(meta, sheet_path)

    assert _pixel_colors(widget, (PetState.ERROR, 1)) == _pixel_colors(widget, (PetState.IDLE, 1))


def test_set_facing_with_same_value_does_not_call_update(
    qapp: QApplication, tmp_path: Path
) -> None:
    sheet_path = tmp_path / "spritesheet.png"
    meta = _make_sheet(sheet_path)
    widget = PetSpriteWidget(meta, sheet_path)
    calls = []
    widget.update = lambda: calls.append(1)  # type: ignore[method-assign]

    widget.set_facing(True)

    assert calls == []


def test_set_facing_false_mirrors_the_rendered_image_horizontally(
    qapp: QApplication, tmp_path: Path
) -> None:
    """set_frame(IDLE, 0) 缓存的是左红右蓝；翻转后实际渲染出的画面应变成左蓝右红。"""
    sheet_path = tmp_path / "spritesheet.png"
    meta = _make_sheet(sheet_path)
    widget = PetSpriteWidget(meta, sheet_path)
    widget.set_frame(PetState.IDLE, 0)

    widget.set_facing(False)

    image = widget.grab().toImage()
    left = image.pixelColor(0, image.height() // 2)
    right = image.pixelColor(image.width() - 1, image.height() // 2)
    assert (left.red(), left.green(), left.blue()) == _BLUE[:3]
    assert (right.red(), right.green(), right.blue()) == _RED[:3]
