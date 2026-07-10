"""右键圆环菜单：右键点击宠物本体后弹出的自绘圆盘，四个操作沿圆周排布。

用 ``Qt.WindowType.Popup`` 而非手写全局事件过滤器来实现"点击外部自动关闭"——这是
``QMenu`` 内部实现同一效果的方式。角度命中判定用 ``atan2(-dy, dx)``（对 dy 取反）把
Qt 的 y-down 屏幕坐标转换成"右为 0°、逆时针递增"的数学惯例角度，与 ``QPainterPath.arcTo``
的角度定义一致，因此 ``_sector_at`` 命中范围和 ``paintEvent`` 里扇区的绘制范围天然对齐，
不需要额外的换算表。
"""

from __future__ import annotations

import math
from enum import Enum, auto

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent, QPainter, QPainterPath, QPaintEvent
from PySide6.QtWidgets import QWidget

from miku_on_desk.face.ui.theme import TEAL_DARK, TEAL_MAIN, qcolor

_OUTER_RADIUS = 90
_INNER_RADIUS = 36
_SIZE = _OUTER_RADIUS * 2


class _Sector(Enum):
    TALK = auto()
    SETTINGS = auto()
    MEMORY = auto()
    RECOLLECTIONS = auto()
    CHARACTERS = auto()
    QUIT = auto()


_SECTOR_ORDER = [
    _Sector.TALK,
    _Sector.SETTINGS,
    _Sector.MEMORY,
    _Sector.RECOLLECTIONS,
    _Sector.CHARACTERS,
    _Sector.QUIT,
]
_SECTOR_LABELS = {
    _Sector.TALK: "对miku说",
    _Sector.SETTINGS: "设置",
    _Sector.MEMORY: "记忆管理",
    _Sector.RECOLLECTIONS: "回忆相册",
    _Sector.CHARACTERS: "角色生成",
    _Sector.QUIT: "退出",
}


def _sector_span_degrees() -> float:
    return 360 / len(_SECTOR_ORDER)


class RadialMenu(QWidget):
    """默认隐藏；调用 ``popup_at`` 定位并显示，点击扇区或按 Esc 后自动关闭。"""

    talk_requested = Signal()
    settings_requested = Signal()
    memory_requested = Signal()
    recollections_requested = Signal()
    characters_requested = Signal()
    quit_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Popup)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.resize(_SIZE, _SIZE)
        self._hovered: _Sector | None = None

    def popup_at(self, global_pos: QPoint) -> None:
        self.move(global_pos.x() - _OUTER_RADIUS, global_pos.y() - _OUTER_RADIUS)
        self.show()

    def _sector_at(self, pos: QPoint) -> _Sector | None:
        center = QPointF(_OUTER_RADIUS, _OUTER_RADIUS)
        delta = QPointF(pos) - center
        distance = math.hypot(delta.x(), delta.y())
        if distance < _INNER_RADIUS or distance > _OUTER_RADIUS:
            return None
        angle = math.degrees(math.atan2(-delta.y(), delta.x())) % 360
        return _SECTOR_ORDER[int(angle // _sector_span_degrees())]

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        hovered = self._sector_at(event.position().toPoint())
        if hovered != self._hovered:
            self._hovered = hovered
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        sector = self._sector_at(event.position().toPoint())
        if sector is not None:
            self._emit_for(sector)
        self.close()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def _emit_for(self, sector: _Sector) -> None:
        if sector is _Sector.TALK:
            self.talk_requested.emit()
        elif sector is _Sector.SETTINGS:
            self.settings_requested.emit()
        elif sector is _Sector.MEMORY:
            self.memory_requested.emit()
        elif sector is _Sector.RECOLLECTIONS:
            self.recollections_requested.emit()
        elif sector is _Sector.CHARACTERS:
            self.characters_requested.emit()
        elif sector is _Sector.QUIT:
            self.quit_requested.emit()

    def _sector_path(self, sector: _Sector) -> QPainterPath:
        start_angle = _SECTOR_ORDER.index(sector) * _sector_span_degrees()
        outer_rect = QRectF(0, 0, _SIZE, _SIZE)
        inner_rect = QRectF(
            _OUTER_RADIUS - _INNER_RADIUS,
            _OUTER_RADIUS - _INNER_RADIUS,
            _INNER_RADIUS * 2,
            _INNER_RADIUS * 2,
        )
        path = QPainterPath()
        path.moveTo(_OUTER_RADIUS, _OUTER_RADIUS)
        path.arcTo(outer_rect, start_angle, _sector_span_degrees())
        path.closeSubpath()
        inner_path = QPainterPath()
        inner_path.addEllipse(inner_rect)
        return path.subtracted(inner_path)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        ring_path = QPainterPath()
        ring_path.setFillRule(Qt.FillRule.OddEvenFill)
        outer_rect = QRectF(0, 0, _SIZE, _SIZE)
        inner_rect = QRectF(
            _OUTER_RADIUS - _INNER_RADIUS,
            _OUTER_RADIUS - _INNER_RADIUS,
            _INNER_RADIUS * 2,
            _INNER_RADIUS * 2,
        )
        ring_path.addEllipse(outer_rect)
        ring_path.addEllipse(inner_rect)
        painter.fillPath(ring_path, qcolor(TEAL_MAIN, alpha=230))

        if self._hovered is not None:
            painter.fillPath(self._sector_path(self._hovered), qcolor(TEAL_DARK, alpha=230))

        pen = painter.pen()
        pen.setColor(qcolor("#ffffff", alpha=150))
        painter.setPen(pen)
        span = _sector_span_degrees()
        for index in range(len(_SECTOR_ORDER)):
            angle = math.radians(index * span)
            painter.drawLine(
                QPointF(_OUTER_RADIUS, _OUTER_RADIUS),
                QPointF(
                    _OUTER_RADIUS + _OUTER_RADIUS * math.cos(angle),
                    _OUTER_RADIUS - _OUTER_RADIUS * math.sin(angle),
                ),
            )

        pen.setColor(qcolor("#ffffff"))
        painter.setPen(pen)
        label_radius = (_OUTER_RADIUS + _INNER_RADIUS) / 2
        for index, sector in enumerate(_SECTOR_ORDER):
            mid_angle = math.radians(index * span + span / 2)
            x = _OUTER_RADIUS + label_radius * math.cos(mid_angle)
            y = _OUTER_RADIUS - label_radius * math.sin(mid_angle)
            rect = QRectF(x - 32, y - 10, 64, 20)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, _SECTOR_LABELS[sector])
