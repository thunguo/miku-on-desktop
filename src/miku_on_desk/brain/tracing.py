"""结构化 JSON 事件跟踪：与人类可读文本日志完全并行、互不派生的诊断数据源。

排查"某次工具调用耗时/某轮 token 用量/fallback 触发链路"这类问题，肉眼翻文本日志效率
太低——这里把同一批埋点事件额外发一份结构化 JSON，写到独立的 ``trace.jsonl``，供脚本化
分析或未来的 eval harness 消费。tracing 是诊断辅助，不是主循环的必要条件，因此
``trace_event`` 自身的任何异常都必须被吞掉，绝不能拖垂调用方。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

TRACE_LOGGER_NAME = "miku_on_desk.trace"

_trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
_fallback_logger = logging.getLogger(__name__)


def trace_event(session_id: str, event: str, **fields: Any) -> None:
    """写一行结构化 JSON 到独立的 trace logger；与人类可读日志完全并行、互不派生。"""
    try:
        payload = {"ts": time.time(), "session_id": session_id, "event": event, **fields}
        _trace_logger.debug(json.dumps(payload, ensure_ascii=False))
    except Exception:
        _fallback_logger.warning("trace_event 写入失败：event=%s", event, exc_info=True)
