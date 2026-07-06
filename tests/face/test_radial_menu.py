"""RadialMenu 的回归测试：角度/半径命中判定与四个信号的路由，不 ``show()`` 弹出窗口。"""

from __future__ import annotations

import math

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from miku_on_desk.face.ui.radial_menu import _INNER_RADIUS, _OUTER_RADIUS, RadialMenu

_MID_RADIUS = (_OUTER_RADIUS + _INNER_RADIUS) / 2


def _point_at(angle_degrees: float, radius: float) -> QPoint:
    angle = math.radians(angle_degrees)
    x = _OUTER_RADIUS + radius * math.cos(angle)
    y = _OUTER_RADIUS - radius * math.sin(angle)
    return QPoint(round(x), round(y))


def _mouse_event(kind: QEvent.Type, pos: QPoint) -> QMouseEvent:
    point = QPointF(pos)
    return QMouseEvent(
        kind,
        point,
        point,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _key_event(key: Qt.Key) -> QKeyEvent:
    return QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (36, "talk"),
        (0, "talk"),
        (71, "talk"),
        (73, "settings"),
        (108, "settings"),
        (145, "memory"),
        (180, "memory"),
        (217, "characters"),
        (252, "characters"),
        (289, "quit"),
        (324, "quit"),
        (359, "quit"),
    ],
)
def test_sector_at_maps_angle_to_expected_sector(
    qapp: QApplication, angle: float, expected: str
) -> None:
    menu = RadialMenu()

    sector = menu._sector_at(_point_at(angle, _MID_RADIUS))

    assert sector is not None
    assert sector.name.lower() == expected


def test_sector_at_returns_none_within_center_hole(qapp: QApplication) -> None:
    menu = RadialMenu()

    assert menu._sector_at(_point_at(45, _INNER_RADIUS - 5)) is None


def test_sector_at_returns_none_outside_ring(qapp: QApplication) -> None:
    menu = RadialMenu()

    assert menu._sector_at(_point_at(45, _OUTER_RADIUS + 5)) is None


def test_click_in_talk_sector_emits_talk_requested(qapp: QApplication) -> None:
    menu = RadialMenu()
    fired: list[None] = []
    menu.talk_requested.connect(lambda: fired.append(None))

    menu.mouseReleaseEvent(
        _mouse_event(QEvent.Type.MouseButtonRelease, _point_at(36, _MID_RADIUS))
    )

    assert fired == [None]


def test_click_in_settings_sector_emits_settings_requested(qapp: QApplication) -> None:
    menu = RadialMenu()
    fired: list[None] = []
    menu.settings_requested.connect(lambda: fired.append(None))

    menu.mouseReleaseEvent(
        _mouse_event(QEvent.Type.MouseButtonRelease, _point_at(108, _MID_RADIUS))
    )

    assert fired == [None]


def test_click_in_memory_sector_emits_memory_requested(qapp: QApplication) -> None:
    menu = RadialMenu()
    fired: list[None] = []
    menu.memory_requested.connect(lambda: fired.append(None))

    menu.mouseReleaseEvent(
        _mouse_event(QEvent.Type.MouseButtonRelease, _point_at(180, _MID_RADIUS))
    )

    assert fired == [None]


def test_click_in_characters_sector_emits_characters_requested(qapp: QApplication) -> None:
    menu = RadialMenu()
    fired: list[None] = []
    menu.characters_requested.connect(lambda: fired.append(None))

    menu.mouseReleaseEvent(
        _mouse_event(QEvent.Type.MouseButtonRelease, _point_at(252, _MID_RADIUS))
    )

    assert fired == [None]


def test_click_in_quit_sector_emits_quit_requested(qapp: QApplication) -> None:
    menu = RadialMenu()
    fired: list[None] = []
    menu.quit_requested.connect(lambda: fired.append(None))

    menu.mouseReleaseEvent(
        _mouse_event(QEvent.Type.MouseButtonRelease, _point_at(324, _MID_RADIUS))
    )

    assert fired == [None]


def test_click_in_center_hole_emits_no_signal(qapp: QApplication) -> None:
    menu = RadialMenu()
    fired: list[None] = []
    menu.talk_requested.connect(lambda: fired.append(None))
    menu.settings_requested.connect(lambda: fired.append(None))
    menu.memory_requested.connect(lambda: fired.append(None))
    menu.characters_requested.connect(lambda: fired.append(None))
    menu.quit_requested.connect(lambda: fired.append(None))

    menu.mouseReleaseEvent(
        _mouse_event(QEvent.Type.MouseButtonRelease, _point_at(45, _INNER_RADIUS - 5))
    )

    assert fired == []


def test_popup_at_positions_widget_centered_on_point(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    menu = RadialMenu()
    monkeypatch.setattr(menu, "show", lambda: None)

    menu.popup_at(QPoint(500, 400))

    assert menu.pos() == QPoint(500 - _OUTER_RADIUS, 400 - _OUTER_RADIUS)


def test_escape_key_closes_menu(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    menu = RadialMenu()
    closed: list[None] = []
    monkeypatch.setattr(menu, "close", lambda: closed.append(None))

    menu.keyPressEvent(_key_event(Qt.Key.Key_Escape))

    assert closed == [None]
