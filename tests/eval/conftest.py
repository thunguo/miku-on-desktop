"""``tests/eval`` 专用 fixture：把 Phase 2 的结构化 trace logger 接到 caplog 上，暴露成一个
可重复取值的 ``list[dict]`` 事件序列，供本包内的回归/能力用例做基于 trace 事件序列的断言，而
不是零散的回调捕获列表。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator

import pytest

from miku_on_desk.brain.tracing import TRACE_LOGGER_NAME


@pytest.fixture
def capture_trace_events(caplog: pytest.LogCaptureFixture) -> Iterator[Callable[[], list[dict]]]:
    """返回一个取值函数而不是列表快照。

    pytest 在同一条测试的不同阶段会整个替换 ``caplog`` 内部的 records 列表，fixture setup 阶段
    拿到的引用到断言阶段可能已经过期；调用方必须在真正需要断言的时刻调用这个函数，触发对
    ``caplog.records`` 的新鲜读取。trace logger 本身 ``propagate=False``（见
    ``brain/tracing.py``），这里临时打开传播，让记录冒泡到 caplog 已挂在根 logger 上的
    handler，测试结束后还原，不影响其他测试。
    """
    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
    original_propagate = trace_logger.propagate
    trace_logger.propagate = True
    caplog.set_level(logging.DEBUG, logger=TRACE_LOGGER_NAME)

    def _events() -> list[dict]:
        return [
            json.loads(record.message)
            for record in caplog.records
            if record.name == TRACE_LOGGER_NAME
        ]

    try:
        yield _events
    finally:
        trace_logger.propagate = original_propagate
