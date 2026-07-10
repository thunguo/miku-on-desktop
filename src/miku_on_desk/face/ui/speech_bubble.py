"""只负责纯 UI 状态与"用户点击了哪个按钮"这一个信号，不知道 bridge/events.py 或 asyncio
的存在——具体是哪次 tool_use 请求确认、把结果按 request_id 送回哪个 Brain 线程，是
overlay_window.py 接线时才关心的事，这样这个 widget 可以脱离 Brain 单独测试和复用。

背景/边框由 ``paintEvent`` 手绘而非 QSS：普通 ``QWidget``（不同于 ``QFrame`` 派生控件）
默认不会套用样式表里的 ``background-color``/``border``，必须显式绘制才会出现——顺带做成
四角切一个斜口的复古像素对话框轮廓，而不是普通圆角矩形。``Antialiasing`` 关闭以保持清晰
的像素颗粒感，与 ``sprite_widget.py`` 的精灵缩放惯例一致。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontDatabase, QPainter, QPainterPath, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from miku_on_desk.face.ui.theme import PINK_ACCENT, TEAL_DARK, TEAL_MAIN, qcolor

_CORNER_NOTCH = 8
_BORDER_WIDTH = 2
_MIN_BUBBLE_HEIGHT = 56
_MAX_BUBBLE_HEIGHT = 240
_CONTENT_MARGIN = 12
_BUTTON_ROW_SPACING = 8

_WIDGET_STYLE = f"""
QLabel {{
    color: #1a1a1a;
    background: transparent;
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QPushButton {{
    background-color: {TEAL_MAIN};
    border: {_BORDER_WIDTH}px solid {TEAL_DARK};
    border-radius: 0px;
    color: #1a1a1a;
    padding: 2px 12px;
}}
QPushButton:hover {{
    background-color: {PINK_ACCENT};
}}
QPushButton:pressed {{
    background-color: {TEAL_DARK};
}}
"""


def _notched_rect_path(width: int, height: int, notch: int) -> QPainterPath:
    path = QPainterPath()
    right = width - 1
    bottom = height - 1
    path.moveTo(notch, 0)
    path.lineTo(right - notch, 0)
    path.lineTo(right, notch)
    path.lineTo(right, bottom - notch)
    path.lineTo(right - notch, bottom)
    path.lineTo(notch, bottom)
    path.lineTo(0, bottom - notch)
    path.lineTo(0, notch)
    path.closeSubpath()
    return path


class SpeechBubble(QWidget):
    """默认隐藏；``show_speech``/``show_confirmation`` 会让它可见，``clear`` 隐藏它。"""

    decision_made = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("speechBubble")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(_WIDGET_STYLE)

        self._label = QLabel()
        self._label.setWordWrap(True)
        self._label.setTextFormat(Qt.TextFormat.PlainText)
        self._label.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self._label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # 文本放进可滚动视口而非直接塞进气泡：气泡高度被夹在 _MAX_BUBBLE_HEIGHT，超出这个
        # 上限的长回复以前会被 QLabel 直接裁掉（且从顶部裁，最新/结尾内容反而看不到）。改成
        # QScrollArea 后超长内容仍可滚动查看，并靠下面的 rangeChanged→贴底逻辑保证流式追加
        # 时始终显示到最后一行。
        self._scroll = QScrollArea(self)
        self._scroll.setWidget(self._label)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.viewport().setAutoFillBackground(False)
        scrollbar = self._scroll.verticalScrollBar()
        # 内容高度变化（流式追加/整段替换）时把滚动条钉到底——展示以“最后的消息”为准。
        # 流式结束后不再触发 rangeChanged，用户可自由上滚回看更早内容。
        scrollbar.rangeChanged.connect(lambda _minimum, maximum: scrollbar.setValue(maximum))

        self._yes_button = QPushButton("是", self)
        self._no_button = QPushButton("否", self)
        self._yes_button.clicked.connect(lambda: self._emit_decision(True))
        self._no_button.clicked.connect(lambda: self._emit_decision(False))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            _CONTENT_MARGIN, _CONTENT_MARGIN, _CONTENT_MARGIN, _CONTENT_MARGIN
        )
        layout.setSpacing(_BUTTON_ROW_SPACING)
        layout.addWidget(self._scroll)
        button_row = QHBoxLayout()
        button_row.addWidget(self._yes_button)
        button_row.addWidget(self._no_button)
        layout.addLayout(button_row)

        self._set_buttons_visible(False)
        self.hide()

    def show_speech(self, text: str) -> None:
        self._set_buttons_visible(False)
        self._label.setText(text)
        self.show()

    def append_speech(self, delta: str) -> None:
        self._set_buttons_visible(False)
        self._label.setText(self._label.text() + delta)
        self.show()

    def show_confirmation(self, question: str) -> None:
        self._label.setText(question)
        self._set_buttons_visible(True)
        self.show()

    def clear(self) -> None:
        self._label.setText("")
        self._set_buttons_visible(False)
        self.hide()

    def current_text(self) -> str:
        return self._label.text()

    def is_awaiting_confirmation(self) -> bool:
        """用 ``isVisibleTo(self)`` 而非 ``isVisible()``：后者会一并考虑祖先窗口是否已
        ``show()``，导致这个气泡被嵌入一个尚未显示的父窗口时永远判定为不可见——这个方法
        只关心气泡自己有没有切到确认态，不应该受宿主窗口显示时机影响。
        """
        return self._yes_button.isVisibleTo(self)

    def ideal_height(self, width: int) -> int:
        """给定气泡宽度，算出容纳当前文本（及确认态按钮行）所需的高度，夹在
        ``[_MIN_BUBBLE_HEIGHT, _MAX_BUBBLE_HEIGHT]`` 之间。纯函数，不修改任何状态，
        调用方（``overlay_window.py``）据此决定要不要连带调整宿主窗口的尺寸。
        """
        label_width = max(width - 2 * _CONTENT_MARGIN, 1)
        text_height = self._label.heightForWidth(label_width)
        if text_height < 0:
            text_height = self._label.sizeHint().height()
        total = 2 * _CONTENT_MARGIN + text_height
        if self.is_awaiting_confirmation():
            total += _BUTTON_ROW_SPACING + self._yes_button.sizeHint().height()
        return max(_MIN_BUBBLE_HEIGHT, min(total, _MAX_BUBBLE_HEIGHT))

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        path = _notched_rect_path(self.width(), self.height(), _CORNER_NOTCH)
        painter.setPen(QPen(qcolor(TEAL_DARK), _BORDER_WIDTH))
        painter.setBrush(qcolor("#fdfdf5", 235))
        painter.drawPath(path)
        painter.end()
        super().paintEvent(event)

    def _set_buttons_visible(self, visible: bool) -> None:
        self._yes_button.setVisible(visible)
        self._no_button.setVisible(visible)

    def _emit_decision(self, approved: bool) -> None:
        self.decision_made.emit(approved)
        self._set_buttons_visible(False)
