"""单轮内工具往复过长时的上下文压缩：``brain/loop.py`` 的 ``compact_context`` 回调契约本身
已经实现并测试过（见 ``tests/brain/test_loop.py``），这里补上一个具体实现，供
``bridge/events.py::build_loop_callbacks()`` 接线。

压缩只发生在同一次 ``run_ai_loop`` 调用内部——工具调用轮次多到 ``working_messages`` 累积
过大时，摘要较早的工具往复，只保留最近若干条消息的原文。摘要落盘为一条情景记忆事件
（``episodic.append_event``），跨轮/跨 session 的检索由情景记忆通道负责，不属于这个模块
的职责。

token 数没有可靠的计数依据（没有引入 tiktoken 之类的分词器依赖）——用"字符数 // 2"这个粗略
估算，偏保守地估高：CJK 场景下更接近真实 token 数，英文场景下会略微高估，可接受，好过假装
精确。

切分点必须落在 assistant 消息上（下标为奇数，因为整轮消息严格从 user 开头交替），这样摘要
（user 角色）插在它前面之后，序列依然满足 user/assistant 严格交替——不满足的话 Anthropic 的
Messages API 会直接拒绝请求。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime

from miku_on_desk.brain.loop import CompactContextCallback
from miku_on_desk.brain.memory.system import MemorySystem
from miku_on_desk.brain.model_router import ModelRouter
from miku_on_desk.brain.providers.base import (
    ImageBlock,
    Message,
    Provider,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from miku_on_desk.config.settings import ModelTier, ProviderName

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


_COMPACTION_TIER = ModelTier.FAST
_CHARS_PER_TOKEN_ESTIMATE = 2
_IMAGE_TOKEN_ESTIMATE = 1600
_DEFAULT_TOKEN_THRESHOLD = 60_000
_DEFAULT_KEEP_RECENT = 6

_COMPACTION_SYSTEM_PROMPT = (
    "你负责把一段过长的工具调用往复过程压缩成一段简短摘要，只保留后续对话还需要的关键信息"
    "（做了什么操作、发现了什么、还没做完什么），丢弃可以安全丢弃的中间细节（如完整文件内容、"
    "冗长的工具原始输出）。直接输出摘要正文，不要输出任何其他文字。"
)


def estimate_tokens(messages: Sequence[Message]) -> int:
    return sum(_estimate_message_tokens(m) for m in messages)


def _estimate_message_tokens(message: Message) -> int:
    if isinstance(message.content, str):
        return len(message.content) // _CHARS_PER_TOKEN_ESTIMATE
    total = 0
    for block in message.content:
        if isinstance(block, TextBlock):
            total += len(block.text) // _CHARS_PER_TOKEN_ESTIMATE
        elif isinstance(block, ToolUseBlock):
            total += (
                len(block.name) + len(json.dumps(block.input))
            ) // _CHARS_PER_TOKEN_ESTIMATE
        elif isinstance(block, ToolResultBlock):
            total += len(block.content) // _CHARS_PER_TOKEN_ESTIMATE
        elif isinstance(block, ImageBlock):
            total += _IMAGE_TOKEN_ESTIMATE
    return total


def _split_for_compaction(
    messages: Sequence[Message], keep_recent: int
) -> tuple[list[Message], list[Message]]:
    keep = min(keep_recent, len(messages))
    cut = len(messages) - keep
    if cut % 2 == 0:
        cut += 1
    cut = min(cut, len(messages))
    return list(messages[:cut]), list(messages[cut:])


def _render_message_for_summary(message: Message) -> str:
    if isinstance(message.content, str):
        return f"{message.role}: {message.content}"
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            parts.append(f"[调用工具 {block.name}] {json.dumps(block.input, ensure_ascii=False)}")
        elif isinstance(block, ToolResultBlock):
            parts.append(f"[工具结果] {block.content}")
        elif isinstance(block, ImageBlock):
            parts.append("[图片内容]")
    return f"{message.role}: " + "\n".join(parts)


def make_compact_context(
    *,
    session_id: str,
    router: ModelRouter,
    providers: dict[ProviderName, Provider],
    memory_system: MemorySystem,
    token_threshold: int = _DEFAULT_TOKEN_THRESHOLD,
    keep_recent: int = _DEFAULT_KEEP_RECENT,
) -> CompactContextCallback:
    async def compact(messages: list[Message]) -> list[Message] | None:
        if estimate_tokens(messages) < token_threshold:
            return None
        older, recent = _split_for_compaction(messages, keep_recent)
        if not older:
            return None

        transcript = "\n\n".join(_render_message_for_summary(m) for m in older)
        try:
            resolved = router.resolve(_COMPACTION_TIER)
            provider = providers[resolved.provider]
            result = await provider.stream(
                model=resolved.model_id,
                system=_COMPACTION_SYSTEM_PROMPT,
                messages=[Message(role="user", content=transcript)],
                tools=[],
            )
        except Exception:
            logger.exception("上下文压缩调用异常，本轮跳过压缩")
            return None
        if not result.success or not result.content:
            return None

        memory_system.episodic.append_event(
            title="[压缩摘要]",
            summary=result.content,
            occurred_at=_now_iso(),
            source_units=[],
            session_id=session_id,
            model=resolved.model_id,
        )
        summary_message = Message(role="user", content=f"[之前工具往复的摘要]\n{result.content}")
        return [summary_message, *recent]

    return compact
