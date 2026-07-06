"""OpenAICompatibleProvider 的单元测试：mock SDK client，验证消息/工具格式转换与分块累积。

Chat Completions 的流式响应把一次 tool_call 拆成多个 chunk（先给 id/name，再逐段给
arguments 的部分 JSON），所以这里的 fake chunk 构造特别覆盖"跨多个 chunk 拼出一次完整
tool_use"的场景，这是与 Anthropic provider 测试最大的不同点。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from miku_on_desk.brain.providers.base import Message, ToolDefinition
from miku_on_desk.brain.providers.openai_compatible_provider import OpenAICompatibleProvider


async def _achunks(chunks: list[Any]) -> Any:
    for chunk in chunks:
        yield chunk


def _provider_with_response(response: Any) -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider(api_key="fake-key")
    provider._client.chat.completions.create = AsyncMock(return_value=response)  # type: ignore[method-assign]
    return provider


def _tool_call_delta(
    index: int, *, id: str | None = None, name: str = "", arguments: str = ""
) -> Any:
    return SimpleNamespace(
        index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def _chunk(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
    usage: Any = None,
    reasoning_content: str | None = None,
) -> Any:
    delta = SimpleNamespace(
        content=content, tool_calls=tool_calls, reasoning_content=reasoning_content
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def _usage(
    prompt_tokens: int = 0, completion_tokens: int = 0, cached_tokens: int | None = None
) -> Any:
    details = SimpleNamespace(cached_tokens=cached_tokens) if cached_tokens is not None else None
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=details,
    )


async def test_stream_dispatches_content_deltas_and_returns_final_content() -> None:
    chunks = [
        _chunk(content="你好"),
        _chunk(content="世界", finish_reason="stop", usage=_usage(10, 5, cached_tokens=2)),
    ]
    provider = _provider_with_response(_achunks(chunks))
    received: list[str] = []

    result = await provider.stream(
        model="gpt-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[],
        on_content=received.append,
    )

    assert received == ["你好", "世界"]
    assert result.success is True
    assert result.content == "你好世界"
    assert result.stop_reason == "end_turn"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.usage.cache_read_input_tokens == 2


async def test_stream_dispatches_vendor_reasoning_content_via_on_thinking() -> None:
    chunks = [
        _chunk(reasoning_content="推理中"),
        _chunk(content="答案", finish_reason="stop"),
    ]
    provider = _provider_with_response(_achunks(chunks))
    thoughts: list[str] = []
    content: list[str] = []

    result = await provider.stream(
        model="gpt-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[],
        on_content=content.append,
        on_thinking=thoughts.append,
    )

    assert thoughts == ["推理中"]
    assert result.reasoning == "推理中"
    assert result.content == "答案"


async def test_stream_accumulates_tool_call_across_multiple_chunks() -> None:
    chunks = [
        _chunk(tool_calls=[_tool_call_delta(0, id="call_1", name="do_thing", arguments="")]),
        _chunk(tool_calls=[_tool_call_delta(0, arguments='{"a":')]),
        _chunk(tool_calls=[_tool_call_delta(0, arguments=" 1}")], finish_reason="tool_calls"),
    ]
    provider = _provider_with_response(_achunks(chunks))

    result = await provider.stream(
        model="gpt-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[ToolDefinition(name="do_thing", description="d", input_schema={})],
    )

    assert result.stop_reason == "tool_use"
    assert len(result.tool_uses) == 1
    assert result.tool_uses[0].id == "call_1"
    assert result.tool_uses[0].name == "do_thing"
    assert result.tool_uses[0].input == {"a": 1}


async def test_stream_returns_idle_timeout_error_result_without_raising() -> None:
    async def _never_ends() -> Any:
        import asyncio

        await asyncio.sleep(10)
        yield _chunk(content="too late")

    provider = _provider_with_response(_never_ends())

    result = await provider.stream(
        model="gpt-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[],
        idle_timeout_s=0.02,
        hard_timeout_s=1.0,
    )

    assert result.success is False
    assert result.error == "request_idle_timeout"


async def test_stream_maps_connection_error_without_status_code_to_connection_error_token() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    error = openai.APIConnectionError(message="boom", request=request)
    provider = _provider_with_response(_achunks([]))
    provider._client.chat.completions.create = AsyncMock(side_effect=error)  # type: ignore[method-assign]

    result = await provider.stream(
        model="gpt-x",
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
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=status_code, request=request)
    error = openai.APIStatusError(message="boom", response=response, body=None)
    provider = _provider_with_response(_achunks([]))
    provider._client.chat.completions.create = AsyncMock(side_effect=error)  # type: ignore[method-assign]

    result = await provider.stream(
        model="gpt-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[],
    )

    assert result.success is False
    assert result.error == expected_token


async def test_stream_splits_tool_result_blocks_into_separate_tool_messages() -> None:
    from miku_on_desk.brain.providers.base import TextBlock, ToolResultBlock

    chunks = [_chunk(content="ok", finish_reason="stop")]
    provider = _provider_with_response(_achunks(chunks))

    await provider.stream(
        model="gpt-x",
        system="系统提示",
        messages=[
            Message(role="user", content="第一条"),
            Message(
                role="user",
                content=[
                    ToolResultBlock(tool_use_id="call_1", content="工具结果"),
                    TextBlock(text="补充说明"),
                ],
            ),
        ],
        tools=[],
    )

    call_kwargs = provider._client.chat.completions.create.call_args.kwargs  # type: ignore[union-attr]
    sent_messages = call_kwargs["messages"]
    assert sent_messages[0] == {"role": "system", "content": "系统提示"}
    assert sent_messages[1] == {"role": "user", "content": "第一条"}
    assert sent_messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": "工具结果"}
    assert sent_messages[3] == {"role": "user", "content": "补充说明"}


@pytest.mark.parametrize(
    "finish_reason,expected",
    [("stop", "end_turn"), ("tool_calls", "tool_use"), ("length", "max_tokens")],
)
async def test_finish_reason_mapping(finish_reason: str, expected: str) -> None:
    chunks = [_chunk(content="ok", finish_reason=finish_reason)]
    provider = _provider_with_response(_achunks(chunks))

    result = await provider.stream(
        model="gpt-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[],
    )

    assert result.stop_reason == expected
