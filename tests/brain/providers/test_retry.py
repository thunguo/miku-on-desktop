"""classify_status_code / stream_with_retry 的回归测试：假 Provider + 假 sleep，不涉及真实网络
或真实等待时间。
"""

from __future__ import annotations

import pytest

from miku_on_desk.brain.providers.base import (
    Message,
    OnContent,
    OnThinking,
    Provider,
    StreamResult,
    ToolDefinition,
)
from miku_on_desk.brain.providers.retry import classify_status_code, stream_with_retry


class _ScriptedProvider(Provider):
    """按顺序返回预先编排好的一串 StreamResult；``emit_content`` 非空时每次调用都无条件先
    触发 on_content，用来验证"已经流出内容就不再重试"这条安全规则，以及回调确实被转发。
    """

    def __init__(self, results: list[StreamResult], *, emit_content: str | None = None) -> None:
        self._results = results
        self._emit_content = emit_content
        self.call_count = 0

    async def stream(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        on_content: OnContent | None = None,
        on_thinking: OnThinking | None = None,
        idle_timeout_s: float = 120.0,
        hard_timeout_s: float = 600.0,
    ) -> StreamResult:
        result = self._results[self.call_count]
        self.call_count += 1
        if self._emit_content is not None and on_content is not None:
            on_content(self._emit_content)
        return result


async def _fake_sleep(_delay: float) -> None:
    return None


def _messages() -> list[Message]:
    return [Message(role="user", content="hi")]


# ── classify_status_code ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (None, "connection_error"),
        (429, "rate_limited"),
        (500, "server_error"),
        (503, "server_error"),
        (400, "client_error"),
        (401, "client_error"),
        (404, "client_error"),
    ],
)
def test_classify_status_code(status_code: int | None, expected: str) -> None:
    assert classify_status_code(status_code) == expected


# ── stream_with_retry ─────────────────────────────────────────────────────


async def test_stream_with_retry_returns_immediately_on_success() -> None:
    provider = _ScriptedProvider([StreamResult(success=True, content="ok")])

    result = await stream_with_retry(
        provider,
        model="m",
        system="s",
        messages=_messages(),
        tools=[],
        sleep=_fake_sleep,
    )

    assert result.success is True
    assert provider.call_count == 1


async def test_stream_with_retry_retries_on_retryable_error_then_succeeds() -> None:
    provider = _ScriptedProvider(
        [
            StreamResult(success=False, error="rate_limited"),
            StreamResult(success=False, error="server_error"),
            StreamResult(success=True, content="ok"),
        ]
    )

    result = await stream_with_retry(
        provider,
        model="m",
        system="s",
        messages=_messages(),
        tools=[],
        max_retries=3,
        sleep=_fake_sleep,
    )

    assert result.success is True
    assert provider.call_count == 3


async def test_stream_with_retry_gives_up_after_max_retries_exhausted() -> None:
    provider = _ScriptedProvider(
        [StreamResult(success=False, error="server_error") for _ in range(10)]
    )

    result = await stream_with_retry(
        provider,
        model="m",
        system="s",
        messages=_messages(),
        tools=[],
        max_retries=2,
        sleep=_fake_sleep,
    )

    assert result.success is False
    assert result.error == "server_error"
    assert provider.call_count == 3


@pytest.mark.parametrize("error_token", ["client_error", "request_idle_timeout"])
async def test_stream_with_retry_does_not_retry_non_retryable_errors(error_token: str) -> None:
    provider = _ScriptedProvider(
        [
            StreamResult(success=False, error=error_token),
            StreamResult(success=True, content="should never be reached"),
        ]
    )

    result = await stream_with_retry(
        provider,
        model="m",
        system="s",
        messages=_messages(),
        tools=[],
        sleep=_fake_sleep,
    )

    assert result.success is False
    assert result.error == error_token
    assert provider.call_count == 1


async def test_stream_with_retry_does_not_retry_once_content_already_emitted() -> None:
    provider = _ScriptedProvider(
        [
            StreamResult(success=False, error="server_error"),
            StreamResult(success=True, content="should never be reached"),
        ],
        emit_content="部分内容",
    )

    result = await stream_with_retry(
        provider,
        model="m",
        system="s",
        messages=_messages(),
        tools=[],
        sleep=_fake_sleep,
    )

    assert result.success is False
    assert provider.call_count == 1


async def test_stream_with_retry_forwards_on_content_to_caller() -> None:
    provider = _ScriptedProvider([StreamResult(success=True, content="ok")], emit_content="你好")
    received: list[str] = []

    result = await stream_with_retry(
        provider,
        model="m",
        system="s",
        messages=_messages(),
        tools=[],
        on_content=received.append,
        sleep=_fake_sleep,
    )

    assert result.success is True
    assert received == ["你好"]


async def test_stream_with_retry_forwards_on_thinking_to_caller() -> None:
    class _ThinkingProvider(Provider):
        async def stream(
            self,
            *,
            model: str,
            system: str,
            messages: list[Message],
            tools: list[ToolDefinition],
            on_content: OnContent | None = None,
            on_thinking: OnThinking | None = None,
            idle_timeout_s: float = 120.0,
            hard_timeout_s: float = 600.0,
        ) -> StreamResult:
            if on_thinking is not None:
                on_thinking("思考中")
            return StreamResult(success=True, content="ok")

    thoughts: list[str] = []

    result = await stream_with_retry(
        _ThinkingProvider(),
        model="m",
        system="s",
        messages=_messages(),
        tools=[],
        on_thinking=thoughts.append,
        sleep=_fake_sleep,
    )

    assert result.success is True
    assert thoughts == ["思考中"]


async def test_stream_with_retry_sleeps_between_retries() -> None:
    provider = _ScriptedProvider(
        [
            StreamResult(success=False, error="rate_limited"),
            StreamResult(success=True, content="ok"),
        ]
    )
    sleep_calls: list[float] = []

    async def _tracking_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    result = await stream_with_retry(
        provider,
        model="m",
        system="s",
        messages=_messages(),
        tools=[],
        sleep=_tracking_sleep,
    )

    assert result.success is True
    assert len(sleep_calls) == 1
    assert sleep_calls[0] > 0

