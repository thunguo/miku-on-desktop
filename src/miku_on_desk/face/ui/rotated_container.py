"""把任意 ``QWidget`` 整体旋转 90 度显示的容器：树莓派 MHS-3.5 屏幕现在用的 fbdev 驱动
固定横向 480x320，X11 层面报不出竖屏分辨率、也不支持 RandR 旋转/换模（``xrandr`` 只有
一个固定 480x320 模式）——要在还没折腾 Phase 0 的 ``dtoverlay=mhs35:rotate=`` 内核级
旋转（会连带需要重新做触摸校准）之前先看到竖屏效果，只能在应用这一层把整个画面转过来。

用 ``QGraphicsView``/``QGraphicsProxyWidget`` 而不是手写 ``QPainter`` 旋转位图：
``QGraphicsProxyWidget`` 会自动把鼠标/触屏事件按同一个变换反向映射回被包裹widget的
本地坐标系，点击位置天然正确，不需要额外手写坐标换算。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QWidget


class RotatedContainer(QGraphicsView):
    """把 ``content`` 顺时针旋转 90 度后铺满整个容器；``content`` 自身完全不知道被转
    过——它继续按照"自己的 ``width()``/``height()`` 就是可用画布"的逻辑正常布局，容器
    只负责在显示层面把渲染结果转个方向。容器自身的尺寸变化（如外层 ``showFullScreen()``
    应用到物理横向分辨率）会被 ``resizeEvent`` 捕获，重新计算 ``content`` 应该被给成
    多大（物理宽高对调）以及旋转后如何居中。
    """

    def __init__(self, content: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background: transparent; border: none;")

        self._content = content
        scene = QGraphicsScene(self)
        self._proxy = scene.addWidget(content)
        self._proxy.setRotation(90)
        self.setScene(scene)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_rotation_layout()

    def _apply_rotation_layout(self) -> None:
        physical_width, physical_height = self.width(), self.height()
        # 旋转 90 度后宽高互换：竖屏内容的"逻辑宽度"对应物理画布的高度，反之亦然。
        logical_width, logical_height = physical_height, physical_width
        self._content.setFixedSize(logical_width, logical_height)
        self._proxy.setTransformOriginPoint(logical_width / 2, logical_height / 2)
        self._proxy.setPos(
            physical_width / 2 - logical_width / 2,
            physical_height / 2 - logical_height / 2,
        )
        self.setSceneRect(0, 0, physical_width, physical_height)
