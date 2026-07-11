"""system.py（`MemorySystem` 门面）的回归测试：remember/recall 契约 + 手动整理三件事。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from miku_on_desk.brain.memory.models import Entity, Fact, MemoryUnit
from miku_on_desk.brain.memory.system import MemorySystem, default_memory_system
from miku_on_desk.config.settings import EnvBootstrap, MemoryTuningConfig


@pytest.fixture
def system(tmp_path: Path) -> MemorySystem:
    return MemorySystem(tmp_path / "memory")


def _unit(
    *,
    session_id: str = "s1",
    role: Literal["user", "assistant", "system"] = "user",
    content: str,
    created_at: str,
) -> MemoryUnit:
    return MemoryUnit(
        id="", session_id=session_id, role=role, content=content, created_at=created_at
    )


def _make_fact(
    *,
    subject: str,
    predicate: str,
    object_: str,
    confidence: float,
    now: str,
    status: Literal["active", "superseded", "conflict", "archived"] = "active",
) -> Fact:
    return Fact(
        id="",
        subject=subject,
        subject_type="person",
        predicate=predicate,
        object=object_,
        object_type="location",
        confidence=confidence,
        source=[],
        valid_from=now,
        recorded_at=now,
        extracted_by="llm:fast",
        status=status,
    )


# ── 装配 ─────────────────────────────────────────────────────────────────


def test_memory_system_creates_four_layer_directories(tmp_path: Path) -> None:
    root = tmp_path / "memory"

    MemorySystem(root)

    assert (root / "base").is_dir()
    assert (root / "semantic").is_dir()
    assert (root / "episodic").is_dir()
    assert (root / "emotional").is_dir()


def test_default_memory_system_uses_explicit_dir_over_bootstrap(tmp_path: Path) -> None:
    explicit_dir = tmp_path / "custom_memory"

    system = default_memory_system(explicit_dir, EnvBootstrap())

    assert system.root == explicit_dir
    assert explicit_dir.is_dir()


def test_default_memory_system_falls_back_to_bootstrap_data_dir(tmp_path: Path) -> None:
    bootstrap = EnvBootstrap(data_dir=tmp_path)

    system = default_memory_system(None, bootstrap)

    assert system.root == tmp_path / "memory"


def test_memory_system_threads_tuning_into_base_and_emotional_stores(tmp_path: Path) -> None:
    tuning = MemoryTuningConfig(base_similarity_threshold=0.42, emotional_confidence_threshold=0.13)

    system = MemorySystem(tmp_path / "memory", tuning=tuning)

    assert system.base._default_similarity_threshold == 0.42
    assert system.emotional.load_preferences()["confidence_threshold"] == 0.13


def test_memory_system_without_explicit_tuning_uses_config_defaults(tmp_path: Path) -> None:
    system = MemorySystem(tmp_path / "memory")

    assert (
        system.base._default_similarity_threshold == MemoryTuningConfig().base_similarity_threshold
    )
    assert (
        system.emotional.load_preferences()["confidence_threshold"]
        == MemoryTuningConfig().emotional_confidence_threshold
    )


def test_tuning_property_returns_constructed_config(tmp_path: Path) -> None:
    tuning = MemoryTuningConfig(retrieval_min_confidence=0.42)

    system = MemorySystem(tmp_path / "memory", tuning=tuning)

    assert system.tuning is tuning


# ── add_memory_unit ──────────────────────────────────────────────────────


def test_add_memory_unit_delegates_to_base_store(system: MemorySystem) -> None:
    unit_id = system.add_memory_unit(_unit(content="你好", created_at="2026-07-06T09:00:00+00:00"))

    loaded = system.base.load(unit_id)
    assert loaded is not None
    assert loaded.content == "你好"


def test_add_memory_unit_writes_similar_units_in_same_session_without_skipping(
    system: MemorySystem,
) -> None:
    first_id = system.add_memory_unit(
        _unit(content="今天天气真好呀，想出去走走。", created_at="2026-07-06T09:00:00+00:00")
    )
    second_id = system.add_memory_unit(
        _unit(content="今天天气真好呀，想出去走走。", created_at="2026-07-06T09:01:00+00:00")
    )

    assert first_id != second_id
    assert system.base.load(first_id) is not None
    assert system.base.load(second_id) is not None


# ── remember/recall ──────────────────────────────────────────────────────


def test_remember_writes_active_fact_with_tool_extracted_by(system: MemorySystem) -> None:
    system.remember("habits/sleep_schedule", "早睡早起")

    facts = system.semantic.list_facts(subject="user", status="active")
    assert len(facts) == 1
    assert facts[0].predicate == "habits/sleep_schedule"
    assert facts[0].object == "早睡早起"
    assert facts[0].extracted_by == "tool:remember"
    assert facts[0].confidence == 1.0


def test_remember_same_key_supersedes_previous_value(system: MemorySystem) -> None:
    system.remember("habits/sleep_schedule", "早睡早起")
    system.remember("habits/sleep_schedule", "熬夜")

    active = system.semantic.list_facts(subject="user", status="active")
    assert len(active) == 1
    assert active[0].object == "熬夜"
    superseded = system.semantic.list_facts(subject="user", status="superseded")
    assert len(superseded) == 1
    assert superseded[0].object == "早睡早起"


def test_recall_finds_remembered_fact_immediately(system: MemorySystem) -> None:
    system.remember("habits/sleep_schedule", "早睡早起")

    hints = system.recall("早睡早起")

    assert any("早睡早起" in hint.text for hint in hints)


def test_recall_falls_back_to_base_units_when_channels_are_short(system: MemorySystem) -> None:
    system.base.append(
        _unit(content="用户提到自己养了一只猫叫小白。", created_at="2026-07-06T09:00:00+00:00")
    )

    hints = system.recall("小白")

    assert any(hint.label == "原始" and "小白" in hint.text for hint in hints)


def test_recall_does_not_exceed_limit_after_base_fallback(system: MemorySystem) -> None:
    for i in range(5):
        system.base.append(
            _unit(content=f"用户提到猫 {i}", created_at=f"2026-07-06T09:0{i}:00+00:00")
        )

    hints = system.recall("猫", limit=2)

    assert len(hints) <= 2


# ── retrieve_hints / retrieve ────────────────────────────────────────────


def test_retrieve_hints_reads_through_to_semantic_layer(system: MemorySystem) -> None:
    system.remember("location", "上海")

    hints = system.retrieve_hints("上海")

    assert any(hint.label == "语义" for hint in hints)


def test_retrieve_assembles_readable_context_block(system: MemorySystem) -> None:
    system.remember("location", "上海")

    text = system.retrieve("上海")

    assert "已知事实" in text


def test_retrieve_uses_configured_min_confidence_threshold(tmp_path: Path) -> None:
    now = "2026-07-06T09:00:00+00:00"
    low_confidence_fact = _make_fact(
        subject="用户", predicate="住在", object_="上海", confidence=0.3, now=now
    )

    strict_system = MemorySystem(
        tmp_path / "strict", tuning=MemoryTuningConfig(retrieval_min_confidence=0.5)
    )
    strict_system.semantic.upsert_fact(low_confidence_fact)
    assert "上海" not in strict_system.retrieve("上海")

    lenient_system = MemorySystem(
        tmp_path / "lenient", tuning=MemoryTuningConfig(retrieval_min_confidence=0.1)
    )
    lenient_system.semantic.upsert_fact(low_confidence_fact)
    assert "上海" in lenient_system.retrieve("上海")


def test_recall_uses_configured_min_confidence_threshold(tmp_path: Path) -> None:
    now = "2026-07-06T09:00:00+00:00"
    low_confidence_fact = _make_fact(
        subject="用户", predicate="住在", object_="上海", confidence=0.3, now=now
    )

    strict_system = MemorySystem(
        tmp_path / "strict", tuning=MemoryTuningConfig(retrieval_min_confidence=0.5)
    )
    strict_system.semantic.upsert_fact(low_confidence_fact)
    assert not any("上海" in hint.text for hint in strict_system.recall("上海"))

    lenient_system = MemorySystem(
        tmp_path / "lenient", tuning=MemoryTuningConfig(retrieval_min_confidence=0.1)
    )
    lenient_system.semantic.upsert_fact(low_confidence_fact)
    assert any("上海" in hint.text for hint in lenient_system.recall("上海"))


def test_retrieve_hints_method_uses_configured_min_confidence_threshold(tmp_path: Path) -> None:
    now = "2026-07-06T09:00:00+00:00"
    low_confidence_fact = _make_fact(
        subject="用户", predicate="住在", object_="上海", confidence=0.3, now=now
    )

    strict_system = MemorySystem(
        tmp_path / "strict", tuning=MemoryTuningConfig(retrieval_min_confidence=0.5)
    )
    strict_system.semantic.upsert_fact(low_confidence_fact)
    assert strict_system.retrieve_hints("上海") == []

    lenient_system = MemorySystem(
        tmp_path / "lenient", tuning=MemoryTuningConfig(retrieval_min_confidence=0.1)
    )
    lenient_system.semantic.upsert_fact(low_confidence_fact)
    assert any(hint.label == "语义" for hint in lenient_system.retrieve_hints("上海"))


# ── run_consolidation ────────────────────────────────────────────────────


def test_run_consolidation_resolves_lingering_conflicts(system: MemorySystem) -> None:
    now = "2026-07-06T09:00:00+00:00"
    system.semantic.upsert_fact(
        _make_fact(subject="用户", predicate="住在", object_="上海", confidence=0.5, now=now)
    )
    system.semantic.upsert_fact(
        _make_fact(subject="用户", predicate="住在", object_="北京", confidence=0.95, now=now)
    )

    system.run_consolidation(now=now)

    active = system.semantic.list_facts(subject="用户", status="active")
    assert len(active) == 1
    assert active[0].object == "北京"


def test_run_consolidation_merges_entities_with_same_name(system: MemorySystem) -> None:
    system.semantic.upsert_entity(
        Entity(
            id="",
            name="Lisa",
            type="person",
            aliases=["莉莎"],
            first_seen="2026-07-01T00:00:00+00:00",
            last_mentioned="2026-07-01T00:00:00+00:00",
            mention_count=2,
        )
    )
    system.semantic.upsert_entity(
        Entity(
            id="",
            name="lisa",
            type="person",
            aliases=[],
            first_seen="2026-07-05T00:00:00+00:00",
            last_mentioned="2026-07-05T00:00:00+00:00",
            mention_count=3,
        )
    )

    system.run_consolidation()

    entities = system.semantic.list_entities()
    assert len(entities) == 1
    assert entities[0].mention_count == 5
    assert "莉莎" in entities[0].aliases


def test_run_consolidation_archives_superseded_facts(system: MemorySystem) -> None:
    now = "2026-07-06T09:00:00+00:00"
    system.semantic.upsert_fact(
        _make_fact(
            subject="用户",
            predicate="住在",
            object_="上海",
            confidence=0.5,
            now=now,
            status="superseded",
        )
    )

    system.run_consolidation()

    archived = system.semantic.list_facts(subject="用户", status="archived")
    assert len(archived) == 1


def test_run_consolidation_updates_last_consolidated_timestamp(system: MemorySystem) -> None:
    system.run_consolidation(now="2026-07-06T09:00:00+00:00")

    assert system.base.get_last_consolidated() == "2026-07-06T09:00:00+00:00"
