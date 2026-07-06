"""共享的 provider 重试/退避封装：把"瞬时性错误自动重试"这件事从 3 个 Provider 实现里
抽出来一份，而不是各写一遍。

只重试一组明确判定为瞬时性的稳定错误 token——``rate_limited`` / ``server_error`` /
``connection_error``。超时（已经等了很久，重试只会雪上加霜）和 ``client_error``
（请求本身就是错的——鉴权/参数问题，重试只会原样重复同一个失败）不重试。

一旦本次尝试里已经有任何内容通过 ``on_content``/``on_thinking`` 流给 UI，就不再重试
本次请求——即便随后这次流式响应整体失败：Gemini 等 Provider 是在处理流式分片的同一个
try/except 里增量调用这两个回调的，失败可能发生在部分可见文本已经进了聊天气泡之后，
重新发起会让可见对话出现重复/错乱内容。
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

from miku_on_desk.brain.providers.base import (
    Message,
    OnContent,
    OnThinking,
    Provider,
    StreamResult,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

RETRYABLE_ERRORS = frozenset({"rate_limited", "server_error", "connection_error"})

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY_S = 1.0
DEFAULT_MAX_DELAY_S = 20.0


def classify_status_code(status_code: int | None) -> str:
    """把 HTTP 状态码归类成稳定错误 token；``None`` 代表连接层面、根本没拿到响应的失败。"""
    if status_code is None:
        return "connection_error"
    if status_code == 429:
        return "rate_limited"
    if status_code >= 500:
        return "server_error"
    return "client_error"


def _backoff_delay(attempt: int, *, base_delay_s: float, max_delay_s: float) -> float:
    exponential = base_delay_s * (2**attempt)
    jittered = exponential * (0.5 + random.random())
    return float(min(jittered, max_delay_s))


async def stream_with_retry(
    provider: Provider,
    *,
    model: str,
    system: str,
    messages: list[Message],
    tools: list[ToolDefinition],
    on_content: OnContent | None = None,
    on_thinking: OnThinking | None = None,
    idle_timeout_s: float = 120.0,
    hard_timeout_s: float = 600.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
    max_delay_s: float = DEFAULT_MAX_DELAY_S,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> StreamResult:
    """对单个 Provider 的 ``stream()`` 做带退避的重试封装，返回形状与 ``Provider.stream()``
    完全一致的 ``StreamResult``，可在调用处直接替换原本的 ``provider.stream(...)``。
    """
    emitted = False

    def _tracking_on_content(text: str) -> None:
        nonlocal emitted
        emitted = True
        if on_content is not None:
            on_content(text)

    def _tracking_on_thinking(text: str) -> None:
        nonlocal emitted
        emitted = True
        if on_thinking is not None:
            on_thinking(text)

    attempt = 0
    while True:
        emitted = False
        result = await provider.stream(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            on_content=_tracking_on_content,
            on_thinking=_tracking_on_thinking,
            idle_timeout_s=idle_timeout_s,
            hard_timeout_s=hard_timeout_s,
        )
        if result.success or emitted or result.error not in RETRYABLE_ERRORS:
            return result
        if attempt >= max_retries:
            return result
        delay = _backoff_delay(attempt, base_delay_s=base_delay_s, max_delay_s=max_delay_s)
        logger.debug(
            "provider 瞬时性错误 %s，第 %d 次重试前等待 %.2f 秒", result.error, attempt + 1, delay
        )
        await sleep(delay)
        attempt += 1
