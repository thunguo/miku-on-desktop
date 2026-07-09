"""trace_event：结构化 JSON 输出格式、异常吞掉行为。

生产环境下 trace logger 的 ``propagate=False`` 是 ``config/logging_config.py::setup_logging``
的装配职责（另有专门测试覆盖），不是 ``trace_event`` 本身的行为——这里测试时强制
``propagate=True`` 绕开该装配，只验证 ``trace_event`` 自身的输出内容与异常安全。

注意：不能在 fixture 里预先把 ``caplog.records`` 存成局部变量再 yield 出去——pytest 在
setup/call 两个测试阶段之间会把 handler 内部的 records 列表整个替换成新对象（而不是原地
clear()），fixture 的 setup 阶段拿到的引用到了 call 阶段（测试体真正跑 ``trace_event`` 的
阶段）就已经是过期的旧列表。必须在测试体内每次都重新读 ``caplog.records`` 属性。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest

from miku_on_desk.brain.tracing import TRACE_LOGGER_NAME, trace_event

_FALLBACK_LOGGER_NAME = "miku_on_desk.brain.tracing"


@pytest.fixture
def _capture_trace(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """靠 propagate=True 让记录冒泡到 caplog 已经挂在根 logger 上的 handler——不要额外把
    caplog.handler 直接挂到 trace logger 上，否则同一条记录会经由"直接挂载"和"冒泡到根"两条
    路径各触发一次 emit，被 caplog 记两遍。"""
    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
    original_propagate = trace_logger.propagate
    trace_logger.propagate = True
    caplog.set_level(logging.DEBUG, logger=TRACE_LOGGER_NAME)
    caplog.set_level(logging.DEBUG, logger=_FALLBACK_LOGGER_NAME)
    try:
        yield caplog
    finally:
        trace_logger.propagate = original_propagate


def _by_logger(records: list[logging.LogRecord], name: str) -> list[logging.LogRecord]:
    return [record for record in records if record.name == name]


def test_trace_event_emits_valid_json_with_required_fields(
    _capture_trace: pytest.LogCaptureFixture,
) -> None:
    trace_event("session-1", "unit_test_event", extra_field=42)

    records = _by_logger(_capture_trace.records, TRACE_LOGGER_NAME)
    assert len(records) == 1
    payload = json.loads(records[0].message)
    assert payload["session_id"] == "session-1"
    assert payload["event"] == "unit_test_event"
    assert payload["extra_field"] == 42
    assert isinstance(payload["ts"], float)


def test_trace_event_swallows_serialization_failures_and_logs_fallback_warning(
    _capture_trace: pytest.LogCaptureFixture,
) -> None:
    class Unserializable:
        pass

    trace_event("session-1", "unit_test_event", bad=Unserializable())

    assert _by_logger(_capture_trace.records, TRACE_LOGGER_NAME) == []
    fallback_records = _by_logger(_capture_trace.records, _FALLBACK_LOGGER_NAME)
    assert len(fallback_records) == 1
    assert "trace_event" in fallback_records[0].message
