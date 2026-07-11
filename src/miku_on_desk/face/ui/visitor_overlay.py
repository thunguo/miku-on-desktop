"""访客路过：非当前角色偶尔弹出打个招呼的纯装饰性小窗口，不驱动任何关系数据、不产生对话。

窗口标志与动画驱动镜像 ``overlay_window.py``（``OverlayWindow``）与
``character_gallery.py``（``CharacterStandTile``）的既有模式,不重新发明；固定时长后
自动关闭,不需要用户交互。
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QVBoxLayout, QWidget

from miku_on_desk.face.sprite_sheet import SpriteSheetMeta, frame_index
from miku_on_desk.face.ui.speech_bubble import SpeechBubble
from miku_on_desk.face.ui.sprite_widget import PetSpriteWidget

_TICK_MS = 33
_BUBBLE_WIDTH = 220
_AUTO_CLOSE_MS = 8000


class VisitorOverlay(QWidget):
    """访客弹窗：一句问候气泡 + 循环播放 idle 动画的精灵,固定时长后自动关闭。"""

    closed = Signal()

    def __init__(
        self,
        pet_dir: Path,
        meta: SpriteSheetMeta,
        greeting: str,
        x: int,
        y: int,
        *,
        scale: float = 1.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = meta.fallback_state
        self._info = meta.states.get(self._state, meta.states[meta.fallback_state])
        self._elapsed_ms = 0

        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowFlags(flags)
        if sys.platform == "darwin":
            self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._bubble = SpeechBubble(self)
        self._bubble.show_speech(greeting)
        self._bubble.setFixedSize(_BUBBLE_WIDTH, self._bubble.ideal_height(_BUBBLE_WIDTH))
        layout.addWidget(self._bubble, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._sprite = PetSpriteWidget(meta, pet_dir / "spritesheet.png", scale=scale, parent=self)
        layout.addWidget(self._sprite, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.move(x, y)
        self.show()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(_TICK_MS)

        QTimer.singleShot(_AUTO_CLOSE_MS, self.close)

    def _on_tick(self) -> None:
        self._elapsed_ms += _TICK_MS
        frame = frame_index(
            self._elapsed_ms / 1000,
            fps=self._info.fps,
            frame_count=self._info.frame_count,
            loop=True,
        )
        self._sprite.set_frame(self._state, frame)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._timer.stop()
        self.closed.emit()
        super().closeEvent(event)
