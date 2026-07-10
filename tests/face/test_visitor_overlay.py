"""``VisitorOverlay`` 的窗口标志、定位、问候展示与自动关闭行为回归测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

from miku_on_desk.face.sprite_sheet import SpriteSheetMeta
from miku_on_desk.face.ui.visitor_overlay import VisitorOverlay


def _make_character_dir(parent: Path, name: str) -> Path:
    pet_dir = parent / name
    pet_dir.mkdir()
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(pet_dir / "spritesheet.png")
    meta = {
        "pet_name": name,
        "frame_width": 4,
        "frame_height": 4,
        "columns": 2,
        "rows": 1,
        "fallback_state": "idle",
        "states": {"idle": {"row": 0, "frame_count": 2, "fps": 10.0, "loop": True}},
    }
    (pet_dir / "pet.json").write_text(json.dumps(meta), encoding="utf-8")
    return pet_dir


def test_visitor_overlay_sets_frameless_tool_and_always_on_top_flags(
    qapp: QApplication, tmp_path: Path
) -> None:
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")

    overlay = VisitorOverlay(pet_dir, meta, "你好呀", 0, 0)

    flags = overlay.windowFlags()
    assert flags & Qt.WindowType.FramelessWindowHint
    assert flags & Qt.WindowType.Tool
    assert flags & Qt.WindowType.WindowStaysOnTopHint


def test_visitor_overlay_sets_decorative_popup_attributes(
    qapp: QApplication, tmp_path: Path
) -> None:
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")

    overlay = VisitorOverlay(pet_dir, meta, "你好呀", 0, 0)

    assert overlay.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating) is True
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose) is True
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground) is True


@pytest.mark.skipif(sys.platform != "darwin", reason="仅 macOS 需要该属性防止失焦自动隐藏")
def test_visitor_overlay_sets_mac_always_show_tool_window_on_darwin(
    qapp: QApplication, tmp_path: Path
) -> None:
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")

    overlay = VisitorOverlay(pet_dir, meta, "你好呀", 0, 0)

    assert overlay.testAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow) is True


def test_visitor_overlay_moves_to_given_coordinates(qapp: QApplication, tmp_path: Path) -> None:
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")

    overlay = VisitorOverlay(pet_dir, meta, "你好呀", 123, 45)

    assert overlay.pos().x() == 123
    assert overlay.pos().y() == 45


def test_visitor_overlay_shows_greeting_in_speech_bubble(
    qapp: QApplication, tmp_path: Path
) -> None:
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")

    overlay = VisitorOverlay(pet_dir, meta, "路过打个招呼～", 0, 0)

    assert overlay._bubble.current_text() == "路过打个招呼～"


def test_visitor_overlay_is_visible_after_construction(qapp: QApplication, tmp_path: Path) -> None:
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")

    overlay = VisitorOverlay(pet_dir, meta, "你好呀", 0, 0)

    assert overlay.isVisible() is True


def test_visitor_overlay_schedules_auto_close_after_8_seconds(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduled: list[int] = []
    monkeypatch.setattr(QTimer, "singleShot", lambda ms, _fn: scheduled.append(ms))
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")

    VisitorOverlay(pet_dir, meta, "你好呀", 0, 0)

    assert scheduled == [8000]


def test_visitor_overlay_auto_close_timer_closes_window_and_emits_closed(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[int, object]] = []
    monkeypatch.setattr(QTimer, "singleShot", lambda ms, fn: captured.append((ms, fn)))
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")
    closed_calls: list[None] = []

    overlay = VisitorOverlay(pet_dir, meta, "你好呀", 0, 0)
    overlay.closed.connect(lambda: closed_calls.append(None))
    captured[0][1]()  # 触发自动关闭回调

    assert closed_calls == [None]
    assert overlay.isVisible() is False


def test_visitor_overlay_close_event_stops_tick_timer(qapp: QApplication, tmp_path: Path) -> None:
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")
    overlay = VisitorOverlay(pet_dir, meta, "你好呀", 0, 0)
    assert overlay._timer.isActive() is True

    overlay.close()

    assert overlay._timer.isActive() is False


def test_visitor_overlay_tick_advances_sprite_frame(qapp: QApplication, tmp_path: Path) -> None:
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")
    overlay = VisitorOverlay(pet_dir, meta, "你好呀", 0, 0)

    initial_key = overlay._sprite._current_key
    # fps=10.0 时每帧需要 100ms，_TICK_MS=33 单次 tick 不足以跨帧，需多次累积
    for _ in range(4):
        overlay._on_tick()

    assert overlay._sprite._current_key != initial_key
