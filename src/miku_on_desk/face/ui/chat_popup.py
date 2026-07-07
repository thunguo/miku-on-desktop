"""聊天输入弹窗：右键圆环菜单"对miku说"或托盘菜单触发的无边框气泡式输入框。

用 ``Qt.WindowType.Tool`` 而非 ``Qt.WindowType.Popup``：后者在 macOS/Windows 上有一个
长期未解决的上游 Qt bug（QTBUG-83490），Popup 窗口在系统层面成不了正常的"key window"，
导致输入法合成拿不到正确的焦点上下文——中文输入法只能把拼音字母原样插入，跳过候选词/
上屏。改用 ``Tool`` 后失去了 Popup 自带的"点外面自动关闭"，靠 ``changeEvent`` 监听
失焦手动补上。
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QWidget

from miku_on_desk.face.ui.theme import RADIUS_XL, TEAL_DARK

_WIDTH = 320
_HEIGHT = 44

_INPUT_STYLE = f"""
QLineEdit {{
    background-color: rgba(143, 218, 198, 200);
    border: 2px solid {TEAL_DARK};
    border-radius: {RADIUS_XL}px;
    padding: 6px 14px;
    color: #1a1a1a;
    font-size: 14px;
}}
"""


class ChatPopup(QWidget):
    """默认隐藏；调用 ``popup_at`` 定位、显示并聚焦输入框。"""

    text_submitted = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(_WIDTH, _HEIGHT)

        self._input = QLineEdit(self)
        self._input.setPlaceholderText("对 Miku 说点什么…")
        self._input.setStyleSheet(_INPUT_STYLE)
        self._input.returnPressed.connect(self._on_submit)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._input)

    def popup_at(self, global_pos: QPoint) -> None:
        self.move(global_pos.x(), global_pos.y())
        self._input.clear()
        self.show()
        self.activateWindow()
        QTimer.singleShot(0, self._input.setFocus)

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow():
            self.close()
            return
        super().changeEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def _on_submit(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self.text_submitted.emit(text)
        self.close()
