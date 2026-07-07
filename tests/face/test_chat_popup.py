"""ChatPopup 的回归测试：回车提交、空文本不提交、Esc 关闭不提交，不 ``show()`` 弹出窗口。"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from miku_on_desk.face.ui.chat_popup import ChatPopup


def _key_event(key: Qt.Key) -> QKeyEvent:
    return QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)


def test_enter_submits_stripped_text(qapp: QApplication) -> None:
    popup = ChatPopup()
    submitted: list[str] = []
    popup.text_submitted.connect(submitted.append)
    popup._input.setText("  你好  ")

    popup._input.returnPressed.emit()

    assert submitted == ["你好"]


def test_enter_with_empty_text_does_not_submit(qapp: QApplication) -> None:
    popup = ChatPopup()
    submitted: list[str] = []
    popup.text_submitted.connect(submitted.append)
    popup._input.setText("   ")

    popup._input.returnPressed.emit()

    assert submitted == []


def test_escape_closes_without_submitting(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup = ChatPopup()
    submitted: list[str] = []
    popup.text_submitted.connect(submitted.append)
    popup._input.setText("没发出去")
    closed: list[None] = []
    monkeypatch.setattr(popup, "close", lambda: closed.append(None))

    popup.keyPressEvent(_key_event(Qt.Key.Key_Escape))

    assert submitted == []
    assert closed == [None]


def test_popup_at_positions_widget_and_clears_previous_text(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup = ChatPopup()
    popup._input.setText("旧文本")
    monkeypatch.setattr(popup, "show", lambda: None)
    monkeypatch.setattr(popup, "activateWindow", lambda: None)

    popup.popup_at(QPoint(200, 300))

    assert popup.pos() == QPoint(200, 300)
    assert popup._input.text() == ""


def test_window_flags_use_tool_not_popup(qapp: QApplication) -> None:
    popup = ChatPopup()

    window_type = popup.windowFlags() & Qt.WindowType.WindowType_Mask

    assert window_type == Qt.WindowType.Tool
    assert window_type != Qt.WindowType.Popup


def test_deactivation_closes_popup(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    popup = ChatPopup()
    closed: list[None] = []
    monkeypatch.setattr(popup, "close", lambda: closed.append(None))
    monkeypatch.setattr(popup, "isActiveWindow", lambda: False)

    popup.changeEvent(QEvent(QEvent.Type.ActivationChange))

    assert closed == [None]


def test_input_has_non_empty_stylesheet(qapp: QApplication) -> None:
    popup = ChatPopup()

    assert popup._input.styleSheet().strip() != ""
