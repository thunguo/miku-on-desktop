"""GeminiProvider 的单元测试：mock SDK client，验证 function_call/thought part 分发与错误映射。

Gemini 的流式响应把每个 chunk 的增量内容放进 ``candidates[0].content.parts``，其中
``function_call`` 已经是完整解析好的 dict（不需要像 OpenAI 那样跨 chunk 拼接部分 JSON），
``thought=True`` 的 text part 对应思考内容——这里的 fake chunk 构造直接镜像这个形状。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from google.genai import errors

from miku_on_desk.brain.providers.base import Message, ToolDefinition
from miku_on_desk.brain.providers.gemini_provider import GeminiProvider


async def _achunks(chunks: list[Any]) -> Any:
    for chunk in chunks:
        yield chunk


def _provider_with_stream(response: Any) -> GeminiProvider:
    provider = GeminiProvider(api_key="fake-key")
    provider._client.aio.models.generate_content_stream = AsyncMock(  # type: ignore[method-assign]
        return_value=response
    )
    return provider


def _text_part(text: str, thought: bool = False) -> Any:
    return SimpleNamespace(function_call=None, text=text, thought=thought)


def _function_call_part(name: str, args: dict[str, Any], call_id: str | None = None) -> Any:
    return SimpleNamespace(
        function_call=SimpleNamespace(id=call_id, name=name, args=args), text=None, thought=None
    )


def _finish_reason(value: str) -> Any:
    return SimpleNamespace(value=value)


def _usage_metadata(
    prompt_token_count: int = 0,
    candidates_token_count: int = 0,
    cached_content_token_count: int = 0,
) -> Any:
    return SimpleNamespace(
        prompt_token_count=prompt_token_count,
        candidates_token_count=candidates_token_count,
        cached_content_token_count=cached_content_token_count,
    )


def _chunk(
    parts: list[Any] | None = None,
    finish_reason: Any = None,
    usage_metadata: Any = None,
) -> Any:
    content = SimpleNamespace(parts=parts) if parts is not None else None
    candidate = SimpleNamespace(content=content, finish_reason=finish_reason)
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage_metadata)


async def test_stream_dispatches_text_parts_and_returns_final_content() -> None:
    chunks = [
        _chunk(parts=[_text_part("你好")]),
        _chunk(
            parts=[_text_part("世界")],
            finish_reason=_finish_reason("STOP"),
            usage_metadata=_usage_metadata(10, 5, cached_content_token_count=2),
        ),
    ]
    provider = _provider_with_stream(_achunks(chunks))
    received: list[str] = []

    result = await provider.stream(
        model="gemini-x",
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


async def test_stream_dispatches_thought_parts_via_on_thinking_separately() -> None:
    chunks = [
        _chunk(parts=[_text_part("推理中", thought=True)]),
        _chunk(parts=[_text_part("答案")], finish_reason=_finish_reason("STOP")),
    ]
    provider = _provider_with_stream(_achunks(chunks))
    thoughts: list[str] = []
    content: list[str] = []

    result = await provider.stream(
        model="gemini-x",
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


async def test_stream_collects_function_call_parts_and_overrides_stop_reason() -> None:
    chunks = [
        _chunk(
            parts=[_function_call_part("do_thing", {"a": 1}, call_id="call_1")],
            finish_reason=_finish_reason("STOP"),
        )
    ]
    provider = _provider_with_stream(_achunks(chunks))

    result = await provider.stream(
        model="gemini-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[ToolDefinition(name="do_thing", description="d", input_schema={})],
    )

    assert result.stop_reason == "tool_use"
    assert len(result.tool_uses) == 1
    assert result.tool_uses[0].id == "call_1"
    assert result.tool_uses[0].name == "do_thing"
    assert result.tool_uses[0].input == {"a": 1}


async def test_stream_synthesizes_call_id_when_sdk_omits_it() -> None:
    chunks = [_chunk(parts=[_function_call_part("do_thing", {}, call_id=None)])]
    provider = _provider_with_stream(_achunks(chunks))

    result = await provider.stream(
        model="gemini-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[ToolDefinition(name="do_thing", description="d", input_schema={})],
    )

    assert result.tool_uses[0].id == "call_0"


async def test_stream_returns_idle_timeout_error_result_without_raising() -> None:
    async def _never_ends() -> Any:
        import asyncio

        await asyncio.sleep(10)
        yield _chunk(parts=[_text_part("too late")])

    provider = _provider_with_stream(_never_ends())

    result = await provider.stream(
        model="gemini-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[],
        idle_timeout_s=0.02,
        hard_timeout_s=1.0,
    )

    assert result.success is False
    assert result.error == "request_idle_timeout"


async def test_stream_maps_server_error_status_code_to_stable_token() -> None:
    error = errors.APIError(code=500, response_json={"error": {"message": "boom"}})
    provider = GeminiProvider(api_key="fake-key")
    provider._client.aio.models.generate_content_stream = AsyncMock(  # type: ignore[method-assign]
        side_effect=error
    )

    result = await provider.stream(
        model="gemini-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[],
    )

    assert result.success is False
    assert result.error == "server_error"
    assert result.raw_error is not None
    assert "boom" in result.raw_error


@pytest.mark.parametrize(
    ("status_code", "expected_token"),
    [(429, "rate_limited"), (400, "client_error")],
)
async def test_stream_maps_client_error_status_codes_to_stable_tokens(
    status_code: int, expected_token: str
) -> None:
    error = errors.ClientError(code=status_code, response_json={"error": {"message": "boom"}})
    provider = GeminiProvider(api_key="fake-key")
    provider._client.aio.models.generate_content_stream = AsyncMock(  # type: ignore[method-assign]
        side_effect=error
    )

    result = await provider.stream(
        model="gemini-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[],
    )

    assert result.success is False
    assert result.error == expected_token


async def test_stream_maps_server_error_subclass_status_code_to_server_error_token() -> None:
    error = errors.ServerError(code=503, response_json={"error": {"message": "boom"}})
    provider = GeminiProvider(api_key="fake-key")
    provider._client.aio.models.generate_content_stream = AsyncMock(  # type: ignore[method-assign]
        side_effect=error
    )

    result = await provider.stream(
        model="gemini-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[],
    )

    assert result.success is False
    assert result.error == "server_error"


@pytest.mark.parametrize(
    "finish_reason,expected", [("STOP", "end_turn"), ("MAX_TOKENS", "max_tokens")]
)
async def test_finish_reason_mapping(finish_reason: str, expected: str) -> None:
    chunks = [_chunk(parts=[_text_part("ok")], finish_reason=_finish_reason(finish_reason))]
    provider = _provider_with_stream(_achunks(chunks))

    result = await provider.stream(
        model="gemini-x",
        system="系统提示",
        messages=[Message(role="user", content="hi")],
        tools=[],
    )

    assert result.stop_reason == expected
