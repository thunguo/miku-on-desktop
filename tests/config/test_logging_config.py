"""setup_logging 对 trace logger 的装配：handler 数量、propagate、重复调用幂等性。"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from miku_on_desk.brain.tracing import TRACE_LOGGER_NAME
from miku_on_desk.config.logging_config import setup_logging


@pytest.fixture
def _restore_logging_state() -> Iterator[None]:
    """setup_logging 会清空并重新装配根 logger 与 trace logger 的 handler 列表——这两个都是
    跨测试共享的全局单例，测试后必须还原，否则会影响同一进程里后续其它测试的日志行为。"""
    root_logger = logging.getLogger()
    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
    original_root_handlers = list(root_logger.handlers)
    original_root_level = root_logger.level
    original_trace_handlers = list(trace_logger.handlers)
    original_trace_level = trace_logger.level
    original_trace_propagate = trace_logger.propagate
    try:
        yield
    finally:
        for handler in root_logger.handlers:
            if handler not in original_root_handlers:
                handler.close()
        for handler in trace_logger.handlers:
            if handler not in original_trace_handlers:
                handler.close()
        root_logger.handlers = original_root_handlers
        root_logger.setLevel(original_root_level)
        trace_logger.handlers = original_trace_handlers
        trace_logger.setLevel(original_trace_level)
        trace_logger.propagate = original_trace_propagate


def test_setup_logging_wires_trace_logger_with_single_non_propagating_handler(
    tmp_path: Path, _restore_logging_state: None
) -> None:
    setup_logging(tmp_path)

    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
    assert len(trace_logger.handlers) == 1
    assert isinstance(trace_logger.handlers[0], RotatingFileHandler)
    assert trace_logger.propagate is False
    assert trace_logger.level == logging.DEBUG


def test_setup_logging_repeated_calls_do_not_duplicate_trace_handlers(
    tmp_path: Path, _restore_logging_state: None
) -> None:
    setup_logging(tmp_path)
    setup_logging(tmp_path)
    setup_logging(tmp_path)

    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
    assert len(trace_logger.handlers) == 1


def test_setup_logging_writes_trace_events_to_dedicated_jsonl_file(
    tmp_path: Path, _restore_logging_state: None
) -> None:
    from miku_on_desk.brain.tracing import trace_event

    setup_logging(tmp_path)
    trace_event("session-1", "unit_test_event")
    for handler in logging.getLogger(TRACE_LOGGER_NAME).handlers:
        handler.flush()

    trace_path = tmp_path / "trace.jsonl"
    assert trace_path.exists()
    assert "unit_test_event" in trace_path.read_text(encoding="utf-8")
