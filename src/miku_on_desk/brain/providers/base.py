"""LLM Provider 的统一抽象：消息/内容块/工具定义/流式结果的跨 Provider 数据模型。

用真正的 ABC + pydantic 模型而不是弱约定的字典对接，原因是本项目要求强类型（`pyproject.toml`
里 mypy strict），裸字典跨层传递会让 loop.py/tools 那一侧完全失去类型检查。

关键设计取舍：工具调用（tool_use）只应该在整个流式响应结束后出现在 ``StreamResult.tool_uses``
里，绝不通过流式回调提前暴露——这是核心安全约束：ai 循环的自动重连/重试逻辑依赖"部分 assistant
消息永不会被提前提交"这一保证，否则一次网络抖动触发的透明重试可能导致同一个工具被执行两次。
``on_content``/``on_thinking`` 回调只用于把展示文本(说话内容/思考过程)实时推给上层 UI，不承载
任何会触发副作用的信息。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    """视觉兜底（screen_analyze）用到的图片输入，data 是 base64 编码的原始字节。"""

    type: Literal["image"] = "image"
    media_type: str
    data: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[ContentBlock]


class ToolDefinition(BaseModel):
    """工具的 provider 无关描述；每个 provider 各自把 input_schema 转换成自己的原生格式。"""

    name: str
    description: str
    input_schema: dict[str, Any]


class StreamUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class StopReason:
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"


class StreamResult(BaseModel):
    """一次流式请求的最终结果；``error`` 是稳定 token（如 ``"request_timeout"``），

    从不直接承载原始异常字符串——上层（UI 的 i18n / 日志）需要能按错误类型分支，而不是
    解析人类可读的英文错误消息。原始异常文本另放 ``raw_error``，只给日志用，不参与任何分支
    判断——``reasoning`` 字段专属模型的 thinking 内容，绝不能拿异常文本顶替。
    """

    success: bool
    content: str = ""
    reasoning: str = ""
    tool_uses: list[ToolUseBlock] = Field(default_factory=list)
    stop_reason: str | None = None
    usage: StreamUsage = Field(default_factory=StreamUsage)
    error: str | None = None
    raw_error: str | None = None


OnContent = Callable[[str], None]
OnThinking = Callable[[str], None]


class Provider(ABC):
    """一份具体实现对应一种 LLM 线路协议（Anthropic 原生 / OpenAI 兼容 / Gemini 原生）。"""

    @abstractmethod
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
        """发起一次流式补全；返回值只在流真正结束（或被两个超时之一打断）后才产出。"""
