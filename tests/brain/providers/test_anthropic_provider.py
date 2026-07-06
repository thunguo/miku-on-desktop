"""AnthropicProvider 的单元测试：mock SDK client，验证格式转换/事件分发/错误映射。

不连真实网络——用一个假的 async context manager 模拟 ``client.messages.stream(...)``
返回的 ``AsyncMessageStreamManager``，事件与 final message 用 ``SimpleNamespace`` 构造，
因为 provider 代码只读取属性，不依赖具体 pydantic 类型。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx
import pytest

from miku_on_desk.brain.providers.anthropic_provider import AnthropicProvider
from miku_on_desk.brain.providers.base import Message, ToolDefinition


class _FakeEventStream:
    def __init__(self, events: list[Any], final_message: Any) -> None:
        self._events = events
        self._final_message = final_message

    def __aiter__(self) -> Any:
        return self._aiter()

    async def _aiter(self) -> Any:
        for event in self._events:
            yield event

    async def get_final_message(self) -> Any:
        return self._final_message


class _FakeStreamManager:
    def __init__(self, stream: Any = None, error: Exception | None = None) -> None:
        self._stream = stream
        self._error = error

    async def __aenter__(self) -> Any:
        if self._error is not None:
            raise self._error
        return self._stream

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False


def _provider_with_stream(manager: _FakeStreamManager) -> AnthropicProvider:
    provider = AnthropicProvider(api_key="fake-key")
    provider._client.messages.stream = MagicMock(return_value=manager)  # type: ignore[method-assign]
    return provider


def _text_delta_event(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=text)
    )


def _thinking_delta_event(thinking: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="thinking_delta", thinking=thinking),
    )


def _final_message(
    content: list[Any], stop_reason: str = "end_turn", **usage_kwargs: int
) -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=usage_kwargs.get("input_tokens", 0),
        output_tokens=usage_kwargs.get("output_tokens", 0),
        cache_creation_input_tokens=usage_kwargs.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=usage_kwargs.get("cache_read_input_tokens", 0),
    )
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage)


async def test_stream_dispatches_text_deltas_and_returns_final_content() -> None:
    events = [_text_delta_event("你好"), _text_delta_event("世界")]
    final = _final_message(
        [SimpleNamespace(type="text", text="你好世界")],
        input_tokens=10,
        output_tokens=5,
    )
    provider = _provider_with_stream(
        _FakeStreamManager(_FakeEventStream(events, final))
    )
    received: list[str] = []

    result = await provider.stream(
        model="claude-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[],
        on_content=received.append,
    )

    assert received == ["你好", "世界"]
    assert result.success is True
    assert result.content == "你好世界"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


async def test_stream_dispatches_thinking_deltas_separately_from_content() -> None:
    events = [_thinking_delta_event("推理中"), _text_delta_event("答案")]
    final = _final_message(
        [
            SimpleNamespace(type="thinking", thinking="推理中"),
            SimpleNamespace(type="text", text="答案"),
        ]
    )
    provider = _provider_with_stream(_FakeStreamManager(_FakeEventStream(events, final)))
    thoughts: list[str] = []
    content: list[str] = []

    result = await provider.stream(
        model="claude-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[],
        on_content=content.append,
        on_thinking=thoughts.append,
    )

    assert thoughts == ["推理中"]
    assert content == ["答案"]
    assert result.reasoning == "推理中"
    assert result.content == "答案"


async def test_stream_collects_tool_use_blocks_from_final_message() -> None:
    final = _final_message(
        [SimpleNamespace(type="tool_use", id="tool_1", name="do_thing", input={"a": 1})],
        stop_reason="tool_use",
    )
    provider = _provider_with_stream(_FakeStreamManager(_FakeEventStream([], final)))

    result = await provider.stream(
        model="claude-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[ToolDefinition(name="do_thing", description="d", input_schema={})],
    )

    assert result.stop_reason == "tool_use"
    assert len(result.tool_uses) == 1
    assert result.tool_uses[0].name == "do_thing"
    assert result.tool_uses[0].input == {"a": 1}


async def test_stream_returns_idle_timeout_error_result_without_raising() -> None:
    async def _never_ends() -> Any:
        import asyncio

        await asyncio.sleep(10)
        yield _text_delta_event("too late")

    fake_stream = SimpleNamespace(
        __aiter__=lambda: _never_ends(),
        get_final_message=AsyncMock(return_value=_final_message([])),
    )
    provider = _provider_with_stream(_FakeStreamManager(fake_stream))

    result = await provider.stream(
        model="claude-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[],
        idle_timeout_s=0.02,
        hard_timeout_s=1.0,
    )

    assert result.success is False
    assert result.error == "request_idle_timeout"


async def test_stream_maps_connection_error_without_status_code_to_connection_error_token() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.APIConnectionError(message="boom", request=request)
    provider = _provider_with_stream(_FakeStreamManager(error=error))

    result = await provider.stream(
        model="claude-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[],
    )

    assert result.success is False
    assert result.error == "connection_error"
    assert result.raw_error is not None
    assert "boom" in result.raw_error


@pytest.mark.parametrize(
    ("status_code", "expected_token"),
    [(429, "rate_limited"), (500, "server_error"), (503, "server_error"), (400, "client_error")],
)
async def test_stream_maps_api_status_error_to_status_code_derived_token(
    status_code: int, expected_token: str
) -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=status_code, request=request)
    error = anthropic.APIStatusError(message="boom", response=response, body=None)
    provider = _provider_with_stream(_FakeStreamManager(error=error))

    result = await provider.stream(
        model="claude-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[],
    )

    assert result.success is False
    assert result.error == expected_token


@pytest.mark.parametrize("role", ["user", "assistant"])
async def test_stream_passes_multi_message_history_through_stream_call(role: str) -> None:
    final = _final_message([SimpleNamespace(type="text", text="ok")])
    provider = _provider_with_stream(_FakeStreamManager(_FakeEventStream([], final)))

    await provider.stream(
        model="claude-x",
        system="系统提示",
        messages=[
            Message(role="user", content="第一条"),
            Message(role=role, content="第二条"),  # type: ignore[arg-type]
        ],
        tools=[],
    )

    call_kwargs = provider._client.messages.stream.call_args.kwargs  # type: ignore[union-attr]
    assert len(call_kwargs["messages"]) == 2
    assert call_kwargs["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }
