"""make_compact_context 的回归测试：假 Provider，不碰真实 LLM。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytest

from miku_on_desk.brain.memory.compaction import (
    _split_for_compaction,
    estimate_tokens,
    make_compact_context,
)
from miku_on_desk.brain.memory.system import MemorySystem
from miku_on_desk.brain.model_router import ModelRouter
from miku_on_desk.brain.providers.base import (
    ImageBlock,
    Message,
    OnContent,
    OnThinking,
    Provider,
    StreamResult,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from miku_on_desk.config.settings import ModelRouterConfig, ModelTier, ProviderConfig, ProviderName


class _FakeProvider(Provider):
    """记录调用参数，按需返回成功或失败的 StreamResult；不真正联网。"""

    def __init__(self, result: StreamResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append({"model": model, "system": system, "messages": list(messages)})
        return self._result


class _RaisingProvider(Provider):
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
        raise RuntimeError("网络炸了")


def _make_router() -> ModelRouter:
    config = ModelRouterConfig()
    config.anthropic = ProviderConfig(
        api_key="sk-ant", models={ModelTier.FAST: "claude-fake-fast"}
    )
    return ModelRouter(config)


def _alternating_messages(count: int) -> list[Message]:
    messages: list[Message] = []
    for i in range(count):
        role: Literal["user", "assistant"] = "user" if i % 2 == 0 else "assistant"
        messages.append(Message(role=role, content=f"消息{i}"))
    return messages


@pytest.fixture
def system(tmp_path: Path) -> MemorySystem:
    return MemorySystem(tmp_path / "memory")


# ── estimate_tokens ──────────────────────────────────────────────────────


def test_estimate_tokens_counts_plain_text_message() -> None:
    messages = [Message(role="user", content="abcd")]

    assert estimate_tokens(messages) == 2


def test_estimate_tokens_counts_tool_use_and_tool_result_blocks() -> None:
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="c1", name="ab", input={})],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="c1", content="abcd")],
        ),
    ]

    assert estimate_tokens(messages) > 0


def test_estimate_tokens_counts_image_block_as_fixed_cost() -> None:
    messages = [
        Message(role="user", content=[ImageBlock(media_type="image/png", data="x")])
    ]

    assert estimate_tokens(messages) == 1600


# ── _split_for_compaction ────────────────────────────────────────────────


def test_split_for_compaction_cut_point_lands_on_assistant_message() -> None:
    messages = _alternating_messages(8)

    older, recent = _split_for_compaction(messages, keep_recent=4)

    assert recent[0].role == "assistant"
    assert older + recent == messages


def test_split_for_compaction_returns_empty_older_for_empty_input() -> None:
    older, recent = _split_for_compaction([], keep_recent=6)

    assert older == []
    assert recent == []


# ── make_compact_context ─────────────────────────────────────────────────


async def test_compact_context_returns_none_when_under_threshold(system: MemorySystem) -> None:
    provider = _FakeProvider(StreamResult(success=True, content="摘要"))
    compact = make_compact_context(
        session_id="s1",
        router=_make_router(),
        providers={ProviderName.ANTHROPIC: provider},
        memory_system=system,
    )

    result = await compact(_alternating_messages(4))

    assert result is None
    assert provider.calls == []


async def test_compact_context_returns_none_for_empty_messages(system: MemorySystem) -> None:
    provider = _FakeProvider(StreamResult(success=True, content="摘要"))
    compact = make_compact_context(
        session_id="s1",
        router=_make_router(),
        providers={ProviderName.ANTHROPIC: provider},
        memory_system=system,
        token_threshold=0,
    )

    assert await compact([]) is None


async def test_compact_context_replaces_older_messages_with_summary(system: MemorySystem) -> None:
    provider = _FakeProvider(StreamResult(success=True, content="这是摘要内容"))
    messages = _alternating_messages(8)
    compact = make_compact_context(
        session_id="s1",
        router=_make_router(),
        providers={ProviderName.ANTHROPIC: provider},
        memory_system=system,
        token_threshold=0,
        keep_recent=4,
    )

    result = await compact(messages)

    assert result is not None
    assert len(result) == 4
    assert result[0].role == "user"
    assert "这是摘要内容" in str(result[0].content)
    assert result[1:] == messages[5:]
    assert "消息0" in provider.calls[0]["messages"][0].content

    events = system.episodic.list_events()
    assert len(events) == 1
    assert events[0].title == "[压缩摘要]"
    assert events[0].summary == "这是摘要内容"
    assert events[0].session_id == "s1"


async def test_compact_context_returns_none_on_provider_failure(system: MemorySystem) -> None:
    provider = _FakeProvider(StreamResult(success=False, error="request_timeout"))
    compact = make_compact_context(
        session_id="s1",
        router=_make_router(),
        providers={ProviderName.ANTHROPIC: provider},
        memory_system=system,
        token_threshold=0,
    )

    result = await compact(_alternating_messages(8))

    assert result is None
    assert system.episodic.list_events() == []


async def test_compact_context_returns_none_on_empty_content(system: MemorySystem) -> None:
    provider = _FakeProvider(StreamResult(success=True, content=""))
    compact = make_compact_context(
        session_id="s1",
        router=_make_router(),
        providers={ProviderName.ANTHROPIC: provider},
        memory_system=system,
        token_threshold=0,
    )

    result = await compact(_alternating_messages(8))

    assert result is None


async def test_compact_context_returns_none_when_provider_raises(system: MemorySystem) -> None:
    compact = make_compact_context(
        session_id="s1",
        router=_make_router(),
        providers={ProviderName.ANTHROPIC: _RaisingProvider()},
        memory_system=system,
        token_threshold=0,
    )

    result = await compact(_alternating_messages(8))

    assert result is None
    assert system.episodic.list_events() == []
