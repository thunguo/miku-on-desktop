"""``RotatedContainer`` 的回归测试：resize 后正确把物理宽高对调成内容应该用的逻辑
尺寸，并把旋转中心对齐回物理画布中心。

``resizeEvent`` 在未 ``show()`` 的 widget 上不会被 Qt 派发，所以测试里直接调
``_apply_rotation_layout()``（``resizeEvent`` 的真正逻辑）而不依赖事件分发本身——
不需要为了触发一次布局计算而真的显示一个窗口。
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QWidget

from miku_on_desk.face.ui.rotated_container import RotatedContainer


def test_resize_swaps_dimensions_for_content_widget(qapp: QApplication) -> None:
    content = QWidget()
    container = RotatedContainer(content)

    container.resize(480, 320)
    container._apply_rotation_layout()

    assert content.width() == 320
    assert content.height() == 480


def test_proxy_rotation_is_90_degrees(qapp: QApplication) -> None:
    content = QWidget()
    container = RotatedContainer(content)

    container.resize(480, 320)
    container._apply_rotation_layout()

    assert container._proxy.rotation() == 90


def test_proxy_transform_origin_is_content_center(qapp: QApplication) -> None:
    content = QWidget()
    container = RotatedContainer(content)

    container.resize(480, 320)
    container._apply_rotation_layout()

    origin = container._proxy.transformOriginPoint()
    assert origin.x() == 160
    assert origin.y() == 240


def test_proxy_position_centers_rotated_content_on_physical_canvas(
    qapp: QApplication,
) -> None:
    content = QWidget()
    container = RotatedContainer(content)

    container.resize(480, 320)
    container._apply_rotation_layout()

    pos = container._proxy.pos()
    assert pos.x() == 480 / 2 - 320 / 2
    assert pos.y() == 320 / 2 - 480 / 2
