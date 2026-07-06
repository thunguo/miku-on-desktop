"""进程级共享的 pytest fixture：QApplication 全进程只能创建一个实例，所有需要实例化
QWidget 的测试（face/ui/ 下）都依赖这个单例，而不是各自零散创建导致重复构造报错。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp() -> Iterator[QApplication]:
    app = QApplication.instance() or QApplication([])
    yield app
