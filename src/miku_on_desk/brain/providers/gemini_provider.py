"""Gemini 原生协议 Provider（``google-genai`` SDK）。

设计要点（与 Anthropic/OpenAI 兼容线路的差异点）：
- ``system_instruction`` 走 ``GenerateContentConfig``，不放进 ``contents`` 列表；Gemini 的
  ``contents`` 只接受 ``user``/``model`` 两种角色，没有独立的 system 消息。
- 不支持显式 ``cache_control`` 断点，理由与 OpenAI 兼容线路相同：没有手动断点接口，依赖服务端
  隐式前缀缓存。
- ``FunctionDeclaration.parameters_json_schema`` 直接接收原始 JSON Schema dict（官方类型标注
  就是 ``Any``），不需要先转换成 Gemini 自己的 ``Schema`` 类型再传。
- ``ToolResultBlock`` 转换到 ``FunctionResponse`` 时只填 ``id``、不填 ``name``：跨 provider 共用
  的 ``ToolResultBlock`` 只携带 ``tool_use_id``，不携带函数名；Gemini 文档说明 ``id`` 有值时按
  ``id`` 匹配对应的 ``function_call``，``name`` 主要用于没有 ``id`` 时的兼容匹配——不为这一个
  provider 的边缘情况扩大所有 provider 共用的 schema。
- ``thinking_config.include_thoughts=True`` 无条件开启：不支持 thinking 的模型会直接忽略这项
  配置，支持的模型才会在 ``Part.thought=True`` 的分片里真正给出推理内容。
- Gemini 的 function_call 参数（``args``）在流式响应里就是已经解析好的 dict，不像 OpenAI 那样
  要跨多个 chunk 拼接部分 JSON 字符串，所以这里不需要 OpenAI provider 里的 ``_ToolCallBuffer``
  等价物。
"""

from __future__ import annotations

import base64
from typing import Any, cast

from google import genai
from google.genai import errors, types

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
    "STOP": StopReason.END_TURN,
    "MAX_TOKENS": StopReason.MAX_TOKENS,
}


def _block_to_gemini_part(block: ContentBlock) -> types.Part:
    if isinstance(block, TextBlock):
        return types.Part(text=block.text)
    if isinstance(block, ImageBlock):
        return types.Part(
            inline_data=types.Blob(mime_type=block.media_type, data=base64.b64decode(block.data))
        )
    if isinstance(block, ToolUseBlock):
        return types.Part(
            function_call=types.FunctionCall(id=block.id, name=block.name, args=block.input)
        )
    if isinstance(block, ToolResultBlock):
        return types.Part(
            function_response=types.FunctionResponse(
                id=block.tool_use_id, response={"output": block.content}
            )
        )
    raise TypeError(f"未知的 ContentBlock 类型：{type(block)!r}")


def _message_to_gemini_content(message: Message) -> types.Content:
    role = "model" if message.role == "assistant" else "user"
    if isinstance(message.content, str):
        return types.Content(role=role, parts=[types.Part(text=message.content)])
    return types.Content(role=role, parts=[_block_to_gemini_part(b) for b in message.content])


def _tools_to_gemini(tools: list[ToolDefinition]) -> list[types.Tool]:
    if not tools:
        return []
    declarations = [
        types.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=tool.input_schema,
        )
        for tool in tools
    ]
    return [types.Tool(function_declarations=declarations)]


class GeminiProvider(Provider):
    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        http_options = types.HttpOptions(base_url=base_url) if base_url else None
        self._client = genai.Client(api_key=api_key, http_options=http_options)

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
        contents = [_message_to_gemini_content(message) for message in messages]
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=cast(Any, _tools_to_gemini(tools)) or None,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )

        content_text = ""
        reasoning_text = ""
        finish_reason: str | None = None
        usage = StreamUsage()
        tool_uses: list[ToolUseBlock] = []
        next_synthetic_id = 0

        try:
            response_stream = await self._client.aio.models.generate_content_stream(
                model=model, contents=cast(Any, contents), config=config
            )
            async for chunk in watch_stream_timeouts(
                response_stream, idle_timeout_s=idle_timeout_s, hard_timeout_s=hard_timeout_s
            ):
                if chunk.usage_metadata is not None:
                    usage = StreamUsage(
                        input_tokens=chunk.usage_metadata.prompt_token_count or 0,
                        output_tokens=chunk.usage_metadata.candidates_token_count or 0,
                        cache_read_input_tokens=(
                            chunk.usage_metadata.cached_content_token_count or 0
                        ),
                    )
                if not chunk.candidates:
                    continue
                candidate = chunk.candidates[0]
                if candidate.finish_reason is not None:
                    finish_reason = candidate.finish_reason.value
                if candidate.content is None or candidate.content.parts is None:
                    continue
                for part in candidate.content.parts:
                    if part.function_call is not None:
                        call_id = part.function_call.id or f"call_{next_synthetic_id}"
                        next_synthetic_id += 1
                        tool_uses.append(
                            ToolUseBlock(
                                id=call_id,
                                name=part.function_call.name or "",
                                input=dict(part.function_call.args or {}),
                            )
                        )
                    elif part.text:
                        if part.thought:
                            reasoning_text += part.text
                            if on_thinking is not None:
                                on_thinking(part.text)
                        else:
                            content_text += part.text
                            if on_content is not None:
                                on_content(part.text)
        except StreamTimeoutError as exc:
            return StreamResult(success=False, error=f"request_{exc.kind}_timeout")
        except errors.APIError as exc:
            status_code = getattr(exc, "code", None)
            return StreamResult(
                success=False, error=classify_status_code(status_code), raw_error=str(exc)
            )

        resolved_stop_reason = (
            StopReason.TOOL_USE
            if tool_uses
            else _FINISH_REASON_MAP.get(finish_reason or "", finish_reason)
        )
        return StreamResult(
            success=True,
            content=content_text,
            reasoning=reasoning_text,
            tool_uses=tool_uses,
            stop_reason=resolved_stop_reason,
            usage=usage,
        )
