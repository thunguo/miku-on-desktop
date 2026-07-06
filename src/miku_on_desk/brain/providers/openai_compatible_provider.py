"""OpenAI 兼容协议 Provider：走 Chat Completions（``/v1/chat/completions``），不用 Responses API。

选 Chat Completions 而非更新的 Responses API，是因为本 provider 的目标不只是 OpenAI 官方
服务，还要覆盖大量"形状兼容 OpenAI"的第三方/自托管端点（各类聚合网关、本地推理服务等）——
这些服务几乎全部实现的是 Chat Completions 形状,Responses API 的覆盖面小得多。

与 Anthropic 不同，这条线路不支持显式 ``cache_control`` 断点：OpenAI 官方走自动前缀缓存
（相同前缀自动复用，无需标注),其他兼容端点即便实现了缓存也各自为政,没有统一的手动断点
接口，因此这里不做任何缓存相关的消息改写。

``reasoning_content`` 不是 Chat Completions 的官方字段，而是部分第三方兼容端点（如 DeepSeek
等）附加的推理内容扩展；openai SDK 的 ``ChoiceDelta`` 模型开了 ``extra="allow"``，未知字段会
保留在实例上而不是被丢弃，所以这里用 ``getattr`` 兜底读取，读不到就是普通不支持推理展示的
端点，静默跳过即可。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast

import openai

from miku_on_desk.brain.providers.base import (
    ContentBlock,
    ImageBlock,
    Message,
    OnContent,
    OnThinking,
    Provider,
    StopReason,
    StreamResult,
    StreamUsage,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from miku_on_desk.brain.providers.retry import classify_status_code
from miku_on_desk.brain.providers.stream_timeout import StreamTimeoutError, watch_stream_timeouts

_FINISH_REASON_MAP = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
}


def _assistant_blocks_to_openai(blocks: list[ContentBlock]) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.input)},
                }
            )
        else:
            raise TypeError(f"assistant 消息不应包含 {type(block)!r}")
    message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _user_blocks_to_openai(blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    """user 消息里的 ToolResultBlock 必须拆成独立的 ``role: tool`` 消息，其余内容保持原样顺序。"""
    messages: list[dict[str, Any]] = []
    buffered: list[dict[str, Any]] = []

    def flush() -> None:
        if not buffered:
            return
        if len(buffered) == 1 and buffered[0]["type"] == "text":
            messages.append({"role": "user", "content": buffered[0]["text"]})
        else:
            messages.append({"role": "user", "content": list(buffered)})
        buffered.clear()

    for block in blocks:
        if isinstance(block, TextBlock):
            buffered.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            buffered.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.media_type};base64,{block.data}"},
                }
            )
        elif isinstance(block, ToolResultBlock):
            flush()
            messages.append(
                {"role": "tool", "tool_call_id": block.tool_use_id, "content": block.content}
            )
        else:
            raise TypeError(f"user 消息不应包含 {type(block)!r}")
    flush()
    return messages


def _message_to_openai(message: Message) -> list[dict[str, Any]]:
    if isinstance(message.content, str):
        return [{"role": message.role, "content": message.content}]
    if message.role == "assistant":
        return [_assistant_blocks_to_openai(message.content)]
    return _user_blocks_to_openai(message.content)


def _messages_to_openai(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        result.extend(_message_to_openai(message))
    return result


def _tools_to_openai(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


@dataclass
class _ToolCallBuffer:
    id: str = ""
    name: str = ""
    arguments_json: str = ""


@dataclass
class _AccumulatedStream:
    content_text: str = ""
    reasoning_text: str = ""
    stop_reason: str | None = None
    usage: StreamUsage = field(default_factory=StreamUsage)
    tool_call_buffers: dict[int, _ToolCallBuffer] = field(default_factory=dict)

    def tool_uses(self) -> list[ToolUseBlock]:
        result: list[ToolUseBlock] = []
        for buffer in self.tool_call_buffers.values():
            try:
                parsed_input = json.loads(buffer.arguments_json) if buffer.arguments_json else {}
            except json.JSONDecodeError:
                parsed_input = {}
            result.append(ToolUseBlock(id=buffer.id, name=buffer.name, input=parsed_input))
        return result


class OpenAICompatibleProvider(Provider):
    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

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
        openai_messages = _messages_to_openai(system, messages)
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": cast(Any, openai_messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            request_kwargs["tools"] = cast(Any, _tools_to_openai(tools))

        acc = _AccumulatedStream()
        try:
            response = await self._client.chat.completions.create(**request_kwargs)
            async for chunk in watch_stream_timeouts(
                response, idle_timeout_s=idle_timeout_s, hard_timeout_s=hard_timeout_s
            ):
                self._consume_chunk(chunk, acc, on_content=on_content, on_thinking=on_thinking)
        except StreamTimeoutError as exc:
            return StreamResult(success=False, error=f"request_{exc.kind}_timeout")
        except openai.APIError as exc:
            status_code = getattr(exc, "status_code", None)
            return StreamResult(
                success=False, error=classify_status_code(status_code), raw_error=str(exc)
            )

        return StreamResult(
            success=True,
            content=acc.content_text,
            reasoning=acc.reasoning_text,
            tool_uses=acc.tool_uses(),
            stop_reason=acc.stop_reason,
            usage=acc.usage,
        )

    @staticmethod
    def _consume_chunk(
        chunk: Any,
        acc: _AccumulatedStream,
        *,
        on_content: OnContent | None,
        on_thinking: OnThinking | None,
    ) -> None:
        if chunk.usage is not None:
            cached_tokens = 0
            if chunk.usage.prompt_tokens_details is not None:
                cached_tokens = chunk.usage.prompt_tokens_details.cached_tokens or 0
            acc.usage = StreamUsage(
                input_tokens=chunk.usage.prompt_tokens,
                output_tokens=chunk.usage.completion_tokens,
                cache_read_input_tokens=cached_tokens,
            )
        if not chunk.choices:
            return
        choice = chunk.choices[0]
        if choice.finish_reason:
            acc.stop_reason = _FINISH_REASON_MAP.get(choice.finish_reason, choice.finish_reason)
        delta = choice.delta
        if delta.content:
            acc.content_text += delta.content
            if on_content is not None:
                on_content(delta.content)
        reasoning_piece = getattr(delta, "reasoning_content", None)
        if reasoning_piece:
            acc.reasoning_text += reasoning_piece
            if on_thinking is not None:
                on_thinking(reasoning_piece)
        if delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                buffer = acc.tool_call_buffers.setdefault(tool_call_delta.index, _ToolCallBuffer())
                if tool_call_delta.id:
                    buffer.id = tool_call_delta.id
                if tool_call_delta.function is not None:
                    if tool_call_delta.function.name:
                        buffer.name += tool_call_delta.function.name
                    if tool_call_delta.function.arguments:
                        buffer.arguments_json += tool_call_delta.function.arguments
