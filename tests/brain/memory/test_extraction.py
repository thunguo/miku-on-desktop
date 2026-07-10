"""extraction.run_extractions 提取流水线的回归测试：假 Provider，不碰真实 LLM。

用一个按 system prompt 关键子串路由的假 Provider，因为语义/情景/情感三路子提取器通过
`asyncio.gather` 并发调用，调用顺序不确定；三个 system prompt 里天然含有互斥的关键词
（"事实三元组"/"情景记忆事件"/"情感/偏好档案"），拿它们路由足以区分三路，不需要引入人工标记。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from miku_on_desk.brain.memory.base_store import BaseStore
from miku_on_desk.brain.memory.emotional_store import EmotionalStore
from miku_on_desk.brain.memory.episodic_store import EpisodicStore
from miku_on_desk.brain.memory.extraction import _parse_emotional_updates, run_extractions
from miku_on_desk.brain.memory.models import MemoryUnit
from miku_on_desk.brain.memory.semantic_store import SemanticStore
from miku_on_desk.brain.model_router import ModelRouter
from miku_on_desk.brain.providers.base import (
    Message,
    OnContent,
    OnThinking,
    Provider,
    StreamResult,
    ToolDefinition,
)
from miku_on_desk.config.settings import ModelRouterConfig, ModelTier, ProviderConfig, ProviderName


def _as_list(
    value: StreamResult | list[StreamResult] | None, *, default: StreamResult
) -> list[StreamResult]:
    if value is None:
        return [default]
    if isinstance(value, StreamResult):
        return [value]
    return list(value)


class _RoutingFakeProvider(Provider):
    """按 system prompt 里的关键子串路由到语义/情景/情感三路各自的结果序列。"""

    def __init__(
        self,
        *,
        semantic: StreamResult | list[StreamResult] | None = None,
        episodic: StreamResult | list[StreamResult] | None = None,
        emotional: StreamResult | list[StreamResult] | None = None,
    ) -> None:
        self._semantic = _as_list(
            semantic, default=StreamResult(success=True, content='{"facts": []}')
        )
        self._episodic = _as_list(
            episodic, default=StreamResult(success=True, content='{"title": "", "summary": ""}')
        )
        self._emotional = _as_list(
            emotional, default=StreamResult(success=True, content='{"updates": []}')
        )
        self._semantic_calls = 0
        self._episodic_calls = 0
        self._emotional_calls = 0
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
        if "事实三元组" in system:
            result = self._semantic[min(self._semantic_calls, len(self._semantic) - 1)]
            self._semantic_calls += 1
            return result
        if "情景记忆事件" in system:
            result = self._episodic[min(self._episodic_calls, len(self._episodic) - 1)]
            self._episodic_calls += 1
            return result
        if "情感/偏好档案" in system:
            result = self._emotional[min(self._emotional_calls, len(self._emotional) - 1)]
            self._emotional_calls += 1
            return result
        raise AssertionError(f"未知的 system prompt：{system}")


def _make_router() -> ModelRouter:
    config = ModelRouterConfig()
    config.anthropic = ProviderConfig(
        api_key="sk-ant", models={ModelTier.FAST: "claude-fake-fast"}
    )
    return ModelRouter(config)


@pytest.fixture
def base(tmp_path: Path) -> BaseStore:
    return BaseStore(tmp_path / "base", index_path=tmp_path / "index.json")


@pytest.fixture
def semantic(tmp_path: Path) -> SemanticStore:
    return SemanticStore(tmp_path / "semantic")


@pytest.fixture
def episodic(tmp_path: Path) -> EpisodicStore:
    return EpisodicStore(tmp_path / "episodic")


@pytest.fixture
def emotional(tmp_path: Path) -> EmotionalStore:
    return EmotionalStore(tmp_path / "emotional")


def _append_turn(
    base: BaseStore, *, session_id: str, occurred_at: str, text: str = "用户提到自己住在上海。"
) -> list[MemoryUnit]:
    user_id = base.append(
        MemoryUnit(id="", session_id=session_id, role="user", content=text, created_at=occurred_at)
    )
    assistant_id = base.append(
        MemoryUnit(
            id="",
            session_id=session_id,
            role="assistant",
            content="好的，记住了。",
            created_at=occurred_at,
        )
    )
    loaded = [base.load(user_id), base.load(assistant_id)]
    return [unit for unit in loaded if unit is not None]


# ── 语义即时触发 ─────────────────────────────────────────────────────────


async def test_run_extractions_writes_semantic_fact_immediately(
    base: BaseStore,
    semantic: SemanticStore,
    episodic: EpisodicStore,
    emotional: EmotionalStore,
    tmp_path: Path,
) -> None:
    units = _append_turn(base, session_id="s1", occurred_at="2026-07-06T09:00:00+00:00")
    provider = _RoutingFakeProvider(
        semantic=StreamResult(
            success=True,
            content=(
                '{"facts": [{"subject": "用户", "subject_type": "person", '
                '"predicate": "住在", "object": "上海", "object_type": "location", '
                '"confidence": 0.9}]}'
            ),
        )
    )
    router = _make_router()

    await run_extractions(
        base=base,
        semantic=semantic,
        episodic=episodic,
        emotional=emotional,
        root=tmp_path,
        session_id="s1",
        units=units,
        router=router,
        providers={ProviderName.ANTHROPIC: provider},
        now="2026-07-06T09:00:00+00:00",
    )

    facts = semantic.list_facts()
    assert len(facts) == 1
    assert facts[0].subject == "用户"
    assert facts[0].predicate == "住在"
    assert facts[0].object == "上海"
    assert facts[0].status == "active"
    assert facts[0].extracted_by == "llm:claude-fake-fast"


async def test_run_extractions_skips_semantic_write_on_provider_failure(
    base: BaseStore,
    semantic: SemanticStore,
    episodic: EpisodicStore,
    emotional: EmotionalStore,
    tmp_path: Path,
) -> None:
    units = _append_turn(base, session_id="s1", occurred_at="2026-07-06T09:00:00+00:00")
    provider = _RoutingFakeProvider(semantic=StreamResult(success=False, error="request_timeout"))
    router = _make_router()

    await run_extractions(
        base=base,
        semantic=semantic,
        episodic=episodic,
        emotional=emotional,
        root=tmp_path,
        session_id="s1",
        units=units,
        router=router,
        providers={ProviderName.ANTHROPIC: provider},
        now="2026-07-06T09:00:00+00:00",
    )

    assert semantic.list_facts() == []


async def test_run_extractions_tolerates_malformed_semantic_json(
    base: BaseStore,
    semantic: SemanticStore,
    episodic: EpisodicStore,
    emotional: EmotionalStore,
    tmp_path: Path,
) -> None:
    units = _append_turn(base, session_id="s1", occurred_at="2026-07-06T09:00:00+00:00")
    provider = _RoutingFakeProvider(semantic=StreamResult(success=True, content="不是 JSON 的文本"))
    router = _make_router()

    await run_extractions(
        base=base,
        semantic=semantic,
        episodic=episodic,
        emotional=emotional,
        root=tmp_path,
        session_id="s1",
        units=units,
        router=router,
        providers={ProviderName.ANTHROPIC: provider},
        now="2026-07-06T09:00:00+00:00",
    )

    assert semantic.list_facts() == []


async def test_run_extractions_resolves_conflicting_facts_across_turns(
    base: BaseStore,
    semantic: SemanticStore,
    episodic: EpisodicStore,
    emotional: EmotionalStore,
    tmp_path: Path,
) -> None:
    router = _make_router()

    units_1 = _append_turn(base, session_id="s1", occurred_at="2026-07-06T09:00:00+00:00")
    provider_1 = _RoutingFakeProvider(
        semantic=StreamResult(
            success=True,
            content=(
                '{"facts": [{"subject": "用户", "subject_type": "person", '
                '"predicate": "住在", "object": "上海", "object_type": "location", '
                '"confidence": 0.5}]}'
            ),
        )
    )
    await run_extractions(
        base=base,
        semantic=semantic,
        episodic=episodic,
        emotional=emotional,
        root=tmp_path,
        session_id="s1",
        units=units_1,
        router=router,
        providers={ProviderName.ANTHROPIC: provider_1},
        now="2026-07-06T09:00:00+00:00",
    )

    units_2 = _append_turn(base, session_id="s1", occurred_at="2026-07-06T09:01:00+00:00")
    provider_2 = _RoutingFakeProvider(
        semantic=StreamResult(
            success=True,
            content=(
                '{"facts": [{"subject": "用户", "subject_type": "person", '
                '"predicate": "住在", "object": "北京", "object_type": "location", '
                '"confidence": 0.95}]}'
            ),
        )
    )
    await run_extractions(
        base=base,
        semantic=semantic,
        episodic=episodic,
        emotional=emotional,
        root=tmp_path,
        session_id="s1",
        units=units_2,
        router=router,
        providers={ProviderName.ANTHROPIC: provider_2},
        now="2026-07-06T09:01:00+00:00",
    )

    active_facts = semantic.list_facts(status="active")
    assert len(active_facts) == 1
    assert active_facts[0].object == "北京"
    superseded_facts = semantic.list_facts(status="superseded")
    assert len(superseded_facts) == 1
    assert superseded_facts[0].object == "上海"


# ── 情感即时触发 ─────────────────────────────────────────────────────────


async def test_run_extractions_merges_emotional_updates_by_dotted_path(
    base: BaseStore,
    semantic: SemanticStore,
    episodic: EpisodicStore,
    emotional: EmotionalStore,
    tmp_path: Path,
) -> None:
    units = _append_turn(base, session_id="s1", occurred_at="2026-07-06T09:00:00+00:00")
    provider = _RoutingFakeProvider(
        emotional=StreamResult(
            success=True,
            content=(
                '{"updates": [{"path": "location_preferences.familiar_cities", '
                '"value": ["上海"], "confidence": 0.9}]}'
            ),
        )
    )
    router = _make_router()

    await run_extractions(
        base=base,
        semantic=semantic,
        episodic=episodic,
        emotional=emotional,
        root=tmp_path,
        session_id="s1",
        units=units,
        router=router,
        providers={ProviderName.ANTHROPIC: provider},
        now="2026-07-06T09:00:00+00:00",
    )

    preferences = emotional.load_preferences()
    assert preferences["location_preferences"]["familiar_cities"] == ["上海"]
    assert preferences["last_updated"] != ""


async def test_run_extractions_skips_emotional_write_when_no_updates(
    base: BaseStore,
    semantic: SemanticStore,
    episodic: EpisodicStore,
    emotional: EmotionalStore,
    tmp_path: Path,
) -> None:
    units = _append_turn(base, session_id="s1", occurred_at="2026-07-06T09:00:00+00:00")
    provider = _RoutingFakeProvider()
    router = _make_router()

    await run_extractions(
        base=base,
        semantic=semantic,
        episodic=episodic,
        emotional=emotional,
        root=tmp_path,
        session_id="s1",
        units=units,
        router=router,
        providers={ProviderName.ANTHROPIC: provider},
        now="2026-07-06T09:00:00+00:00",
    )

    assert emotional.load_preferences()["last_updated"] == ""


async def test_run_extractions_writes_only_updates_meeting_confidence_threshold(
    base: BaseStore,
    semantic: SemanticStore,
    episodic: EpisodicStore,
    emotional: EmotionalStore,
    tmp_path: Path,
) -> None:
    units = _append_turn(base, session_id="s1", occurred_at="2026-07-06T09:00:00+00:00")
    provider = _RoutingFakeProvider(
        emotional=StreamResult(
            success=True,
            content=(
                '{"updates": ['
                '{"path": "location_preferences.familiar_cities", "value": ["上海"], '
                '"confidence": 0.9}, '
                '{"path": "habits.sleep_schedule", "value": "熬夜", "confidence": 0.3}'
                "]}"
            ),
        )
    )
    router = _make_router()

    await run_extractions(
        base=base,
        semantic=semantic,
        episodic=episodic,
        emotional=emotional,
        root=tmp_path,
        session_id="s1",
        units=units,
        router=router,
        providers={ProviderName.ANTHROPIC: provider},
        now="2026-07-06T09:00:00+00:00",
        emotional_confidence_threshold=0.75,
    )

    preferences = emotional.load_preferences()
    assert preferences["location_preferences"]["familiar_cities"] == ["上海"]
    assert "habits" not in preferences


def test_parse_emotional_updates_rejects_legacy_dict_format() -> None:
    legacy = '{"updates": {"location_preferences.familiar_cities": ["上海"]}}'

    assert _parse_emotional_updates(legacy) == []


# ── 情景批量触发 ─────────────────────────────────────────────────────────


async def test_run_extractions_does_not_flush_episodic_before_threshold(
    base: BaseStore,
    semantic: SemanticStore,
    episodic: EpisodicStore,
    emotional: EmotionalStore,
    tmp_path: Path,
) -> None:
    units = _append_turn(base, session_id="s1", occurred_at="2026-07-06T09:00:00+00:00")
    provider = _RoutingFakeProvider()
    router = _make_router()

    await run_extractions(
        base=base,
        semantic=semantic,
        episodic=episodic,
        emotional=emotional,
        root=tmp_path,
        session_id="s1",
        units=units,
        router=router,
        providers={ProviderName.ANTHROPIC: provider},
        now="2026-07-06T09:00:00+00:00",
    )

    assert episodic.list_events() == []
    pending_path = tmp_path / ".tmp" / "pending_extractions.jsonl"
    assert len(pending_path.read_text(encoding="utf-8").splitlines()) == 1


async def test_run_extractions_flushes_episodic_after_six_turns(
    base: BaseStore,
    semantic: SemanticStore,
    episodic: EpisodicStore,
    emotional: EmotionalStore,
    tmp_path: Path,
) -> None:
    provider = _RoutingFakeProvider(
        episodic=StreamResult(
            success=True,
            content=(
                '{"title": "近期动态", "summary": "用户多次提到搬家。", '
                '"emotion_tag": "温暖", "participants": [], "event_chain": []}'
            ),
        )
    )
    router = _make_router()

    all_unit_ids: list[str] = []
    for turn in range(6):
        occurred_at = f"2026-07-06T09:0{turn}:00+00:00"
        units = _append_turn(base, session_id="s1", occurred_at=occurred_at)
        all_unit_ids.extend(unit.id for unit in units)
        await run_extractions(
            base=base,
            semantic=semantic,
            episodic=episodic,
            emotional=emotional,
            root=tmp_path,
            session_id="s1",
            units=units,
            router=router,
            providers={ProviderName.ANTHROPIC: provider},
            now=occurred_at,
        )

    events = episodic.list_events()
    assert len(events) == 1
    assert events[0].title == "近期动态"
    assert events[0].source_units == all_unit_ids
    pending_path = tmp_path / ".tmp" / "pending_extractions.jsonl"
    assert pending_path.read_text(encoding="utf-8").strip() == ""


async def test_run_extractions_flushes_episodic_after_time_threshold(
    base: BaseStore,
    semantic: SemanticStore,
    episodic: EpisodicStore,
    emotional: EmotionalStore,
    tmp_path: Path,
) -> None:
    provider = _RoutingFakeProvider(
        episodic=StreamResult(success=True, content='{"title": "标题", "summary": "摘要"}')
    )
    router = _make_router()

    units_1 = _append_turn(base, session_id="s1", occurred_at="2026-07-06T09:00:00+00:00")
    await run_extractions(
        base=base,
        semantic=semantic,
        episodic=episodic,
        emotional=emotional,
        root=tmp_path,
        session_id="s1",
        units=units_1,
        router=router,
        providers={ProviderName.ANTHROPIC: provider},
        now="2026-07-06T09:00:00+00:00",
    )

    units_2 = _append_turn(base, session_id="s1", occurred_at="2026-07-06T09:11:00+00:00")
    await run_extractions(
        base=base,
        semantic=semantic,
        episodic=episodic,
        emotional=emotional,
        root=tmp_path,
        session_id="s1",
        units=units_2,
        router=router,
        providers={ProviderName.ANTHROPIC: provider},
        now="2026-07-06T09:11:00+00:00",
    )

    events = episodic.list_events()
    assert len(events) == 1
    assert events[0].title == "标题"


# ── 边界 ─────────────────────────────────────────────────────────────────


async def test_run_extractions_is_noop_for_empty_units(
    base: BaseStore,
    semantic: SemanticStore,
    episodic: EpisodicStore,
    emotional: EmotionalStore,
    tmp_path: Path,
) -> None:
    provider = _RoutingFakeProvider()
    router = _make_router()

    await run_extractions(
        base=base,
        semantic=semantic,
        episodic=episodic,
        emotional=emotional,
        root=tmp_path,
        session_id="s1",
        units=[],
        router=router,
        providers={ProviderName.ANTHROPIC: provider},
        now="2026-07-06T09:00:00+00:00",
    )

    assert provider.calls == []
    assert not (tmp_path / ".tmp" / "pending_extractions.jsonl").exists()
