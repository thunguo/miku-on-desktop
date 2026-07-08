"""桌宠精灵图渲染：把 spritesheet 按 `(state, frame)` 预裁剪、放大缓存成 `QPixmap`。

只负责"画出给定的 (state, frame)"这一件事，不知道时间、状态机或事件总线的存在——
`overlay_window.py` 里的定时器算出当前该显示哪一帧后调用 `set_frame`，这样这个 widget
可以脱离 Brain/Hook 单独测试。放大一律用最近邻（`FastTransformation`），绝不能用
`SmoothTransformation`：生成的像素画依赖清晰的方块颗粒感，平滑缩放会把这些棱角重新糊成
模糊的渐变，等于前功尽弃。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import QWidget

from miku_on_desk.face.pet_state import PetState
from miku_on_desk.face.sprite_sheet import SpriteSheetMeta, cell_rect


class PetSpriteWidget(QWidget):
    """展示 spritesheet 中某一帧的静态图像；帧的推进由外部调用 `set_frame` 驱动。"""

    def __init__(
        self,
        meta: SpriteSheetMeta,
        sheet_path: Path,
        *,
        scale: float = 1.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._meta = meta
        self._display_width = max(1, round(meta.frame_width * scale))
        self._display_height = max(1, round(meta.frame_height * scale))

        sheet = QPixmap(str(sheet_path))
        self._frames: dict[tuple[PetState, int], QPixmap] = {}
        # 遍历全部 PetState（而非 meta.states.keys()），让缺行的状态也能通过
        # fallback_state 的行缓存出对应帧，set_frame 就不必再单独处理缺行情况。
        for state in PetState:
            info = meta.states.get(state, meta.states[meta.fallback_state])
            for frame in range(info.frame_count):
                rect = cell_rect(meta, state, frame)
                cell = sheet.copy(rect.x, rect.y, rect.width, rect.height)
                self._frames[(state, frame)] = cell.scaled(
                    self._display_width,
                    self._display_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )

        self.setFixedSize(self._display_width, self._display_height)
        self._current_key = (meta.fallback_state, 0)
        self._current = self._frames[self._current_key]
        self._facing_right = True

    def set_frame(self, state: PetState, frame: int) -> None:
        key = (state, frame)
        if key == self._current_key:
            return
        pixmap = self._frames.get(key)
        if pixmap is None:
            return
        self._current_key = key
        self._current = pixmap
        self.update()

    def set_facing(self, facing_right: bool) -> None:
        if facing_right == self._facing_right:
            return
        self._facing_right = facing_right
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        if not self._facing_right:
            painter.translate(self.width(), 0)
            painter.scale(-1, 1)
        painter.drawPixmap(0, 0, self._current)
