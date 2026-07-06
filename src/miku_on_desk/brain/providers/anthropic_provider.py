"""Anthropic 原生协议 Provider：唯一支持显式 ``cache_control`` 断点的一条线路。

Gemini 与 OpenAI 兼容协议依赖各自服务端的自动前缀缓存，不需要（也不支持）手动打断点；
Anthropic 则要求调用方显式标注最多 4 个 ``cache_control: {"type": "ephemeral"}`` 断点。
这里的断点策略针对 Anthropic prompt cache 做了分工：系统提示本身占一个断点（冻结不变，
命中率最高）；工具定义列表末尾占一个断点（同一会话内工具集基本不变）；最近
``_MESSAGE_CACHE_MARKERS`` 条非系统消息各占一个断点，随对话推进滚动前移，换取"新增一轮
对话只需重新计费最后几条消息"而不是整段历史重新计价。

流被 abort/timeout 打断时不尝试拼出目前为止收到的部分 tool_use JSON 参数去"抢救"一次不完整
但可能可用的工具调用：本 provider 用 SDK 的 ``get_final_message()`` 取最终结果，而
``StreamTimeoutError`` 触发时底层 HTTP 流的消费状态是不确定的，此时再调用
``get_final_message()`` 去抢救部分内容没有可靠性保证，静默返回一个可能损坏的工具调用比
明确报错更危险。
"""

from __future__ import annotations

from typing import Any, cast

import anthropic

from miku_on_desk.brain.providers.base import (
    ContentBlock,
    ImageBlock,
    Message,
    OnContent,
    OnThinking,
    Provider,
    StreamResult,
    StreamUsage,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from miku_on_desk.brain.providers.retry import classify_status_code
from miku_on_desk.brain.providers.stream_timeout import StreamTimeoutError, watch_stream_timeouts

_MESSAGE_CACHE_MARKERS = 2
_MAX_TOKENS = 8192


def _block_to_anthropic(block: ContentBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageBlock):
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": block.media_type, "data": block.data},
        }
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    raise TypeError(f"未知的 ContentBlock 类型：{type(block)!r}")


def _message_to_anthropic(message: Message) -> dict[str, Any]:
    if isinstance(message.content, str):
        content: Any = message.content
    else:
        content = [_block_to_anthropic(block) for block in message.content]
    return {"role": message.role, "content": content}


def _with_message_cache_control(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """在最近 N 条消息的最后一个内容块上打缓存断点；不修改传入的原始字典。"""
    marked_count = 0
    result: list[dict[str, Any]] = []
    for raw in reversed(messages):
        message = dict(raw)
        if marked_count < _MESSAGE_CACHE_MARKERS:
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            else:
                content = [dict(block) for block in content]
            if content:
                content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
            message["content"] = content
            marked_count += 1
        result.append(message)
    result.reverse()
    return result


def _tools_to_anthropic(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    converted = [
        {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
        for tool in tools
    ]
    if converted:
        converted[-1] = {**converted[-1], "cache_control": {"type": "ephemeral"}}
    return converted


class AnthropicProvider(Provider):
    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)

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
        system_blocks = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        anthropic_messages = _with_message_cache_control(
            [_message_to_anthropic(message) for message in messages]
        )
        anthropic_tools = _tools_to_anthropic(tools)

        try:
            # SDK 的 system/messages/tools 形参类型是精确的 TypedDict 联合；我们的转换函数
            # 产出结构上等价但标注为 dict[str, Any] 的字典，mypy 无法结构化匹配，用 cast 跨过。
            async with self._client.messages.stream(
                model=model,
                max_tokens=_MAX_TOKENS,
                system=cast(Any, system_blocks),
                messages=cast(Any, anthropic_messages),
                tools=cast(Any, anthropic_tools),
            ) as stream:
                async for event in watch_stream_timeouts(
                    stream, idle_timeout_s=idle_timeout_s, hard_timeout_s=hard_timeout_s
                ):
                    if event.type != "content_block_delta":
                        continue
                    delta = event.delta
                    if delta.type == "text_delta" and on_content is not None:
                        on_content(delta.text)
                    elif delta.type == "thinking_delta" and on_thinking is not None:
                        on_thinking(delta.thinking)

                final_message = await stream.get_final_message()
        except StreamTimeoutError as exc:
            return StreamResult(success=False, error=f"request_{exc.kind}_timeout")
        except anthropic.APIError as exc:
            status_code = getattr(exc, "status_code", None)
            return StreamResult(
                success=False, error=classify_status_code(status_code), raw_error=str(exc)
            )

        content_text = ""
        reasoning_text = ""
        tool_uses: list[ToolUseBlock] = []
        for block in final_message.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "thinking":
                reasoning_text += block.thinking
            elif block.type == "tool_use":
                tool_uses.append(
                    ToolUseBlock(id=block.id, name=block.name, input=dict(block.input))
                )

        usage = final_message.usage
        return StreamResult(
            success=True,
            content=content_text,
            reasoning=reasoning_text,
            tool_uses=tool_uses,
            stop_reason=final_message.stop_reason,
            usage=StreamUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
                cache_read_input_tokens=usage.cache_read_input_tokens or 0,
            ),
        )
