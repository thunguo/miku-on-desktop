"""conflict.py（语义事实冲突检测/消解）的回归测试，覆盖设计文档 §5.2/§7.3 伪代码场景。"""

from __future__ import annotations

from pathlib import Path

import pytest

from miku_on_desk.brain.memory.conflict import detect_conflicts, resolve_conflicts
from miku_on_desk.brain.memory.models import Fact
from miku_on_desk.brain.memory.semantic_store import SemanticStore


@pytest.fixture
def semantic(tmp_path: Path) -> SemanticStore:
    return SemanticStore(tmp_path / "semantic")


def _fact(
    *,
    fact_id: str,
    subject: str = "用户",
    predicate: str = "喜欢",
    object_: str = "猫",
    confidence: float = 0.9,
    status: str = "active",
    valid_from: str = "2026-07-06T09:00:00+00:00",
) -> Fact:
    return Fact(
        id=fact_id,
        subject=subject,
        subject_type="person",
        predicate=predicate,
        object=object_,
        object_type="concept",
        confidence=confidence,
        source=[],
        valid_from=valid_from,
        recorded_at=valid_from,
        extracted_by="llm:fast",
        status=status,  # type: ignore[arg-type]
    )


# ── detect_conflicts ─────────────────────────────────────────────────────


def test_detect_conflicts_flags_value_conflict_for_same_subject_predicate_different_object() -> (
    None
):
    existing = _fact(fact_id="f-1", predicate="住在", object_="上海")
    new_fact = _fact(fact_id="f-2", predicate="住在", object_="北京")

    conflicts = detect_conflicts(new_fact, [existing])

    value_conflicts = [c for c in conflicts if c.type == "value_conflict"]
    assert len(value_conflicts) == 1
    assert value_conflicts[0].existing.id == "f-1"
    assert value_conflicts[0].resolution == "higher_confidence_wins"


def test_detect_conflicts_no_value_conflict_when_object_matches() -> None:
    existing = _fact(fact_id="f-1", predicate="住在", object_="上海")
    new_fact = _fact(fact_id="f-2", predicate="住在", object_="上海")

    conflicts = detect_conflicts(new_fact, [existing])

    assert [c for c in conflicts if c.type == "value_conflict"] == []


def test_detect_conflicts_no_conflict_for_different_subject() -> None:
    existing = _fact(fact_id="f-1", subject="Lisa", predicate="住在", object_="上海")
    new_fact = _fact(fact_id="f-2", subject="用户", predicate="住在", object_="北京")

    assert detect_conflicts(new_fact, [existing]) == []


def test_detect_conflicts_ignores_superseded_existing_facts() -> None:
    existing = _fact(fact_id="f-1", predicate="住在", object_="上海", status="superseded")
    new_fact = _fact(fact_id="f-2", predicate="住在", object_="北京")

    assert detect_conflicts(new_fact, [existing]) == []


def test_detect_conflicts_flags_temporal_conflict_for_temporal_predicate() -> None:
    existing = _fact(
        fact_id="f-1",
        predicate="工作于",
        object_="旧公司",
        valid_from="2026-01-01T00:00:00+00:00",
    )
    new_fact = _fact(
        fact_id="f-2",
        predicate="工作于",
        object_="新公司",
        valid_from="2026-07-06T00:00:00+00:00",
    )

    conflicts = detect_conflicts(new_fact, [existing])

    temporal_conflicts = [c for c in conflicts if c.type == "temporal_conflict"]
    assert len(temporal_conflicts) == 1
    assert temporal_conflicts[0].resolution == "latest_valid_wins"


def test_detect_conflicts_no_temporal_conflict_for_non_temporal_predicate() -> None:
    existing = _fact(
        fact_id="f-1",
        predicate="喜欢",
        object_="猫",
        valid_from="2026-01-01T00:00:00+00:00",
    )
    new_fact = _fact(
        fact_id="f-2",
        predicate="喜欢",
        object_="狗",
        valid_from="2026-07-06T00:00:00+00:00",
    )

    conflicts = detect_conflicts(new_fact, [existing])

    assert [c for c in conflicts if c.type == "temporal_conflict"] == []


def test_detect_conflicts_skips_self_comparison() -> None:
    fact = _fact(fact_id="f-1", predicate="住在", object_="上海")

    assert detect_conflicts(fact, [fact]) == []


# ── resolve_conflicts ────────────────────────────────────────────────────


def test_resolve_conflicts_passes_through_singleton_groups(semantic: SemanticStore) -> None:
    fact = _fact(fact_id="f-1")

    resolved = resolve_conflicts([fact], semantic=semantic)

    assert resolved == [fact]


def test_resolve_conflicts_picks_higher_confidence_and_marks_loser_superseded(
    semantic: SemanticStore,
) -> None:
    semantic.upsert_fact(_fact(fact_id="f-1", object_="上海", confidence=0.6))
    weak = _fact(fact_id="f-1", object_="上海", confidence=0.6)
    strong = _fact(fact_id="f-2", object_="北京", confidence=0.95)

    resolved = resolve_conflicts([weak, strong], semantic=semantic)

    assert [fact.id for fact in resolved] == ["f-2"]
    assert semantic.get_fact("f-1").status == "superseded"  # type: ignore[union-attr]


def test_resolve_conflicts_prefers_latest_valid_from_when_confidence_is_close(
    semantic: SemanticStore,
) -> None:
    semantic.upsert_fact(
        _fact(
            fact_id="f-1",
            object_="旧公司",
            confidence=0.90,
            valid_from="2026-01-01T00:00:00+00:00",
        )
    )
    older = _fact(
        fact_id="f-1", object_="旧公司", confidence=0.90, valid_from="2026-01-01T00:00:00+00:00"
    )
    newer = _fact(
        fact_id="f-2", object_="新公司", confidence=0.92, valid_from="2026-07-06T00:00:00+00:00"
    )

    resolved = resolve_conflicts([older, newer], semantic=semantic)

    assert [fact.id for fact in resolved] == ["f-2"]
    assert semantic.get_fact("f-1").status == "superseded"  # type: ignore[union-attr]


def test_resolve_conflicts_handles_independent_groups_separately(
    semantic: SemanticStore,
) -> None:
    a1 = _fact(fact_id="f-1", subject="用户", predicate="住在", object_="上海", confidence=0.9)
    a2 = _fact(fact_id="f-2", subject="用户", predicate="住在", object_="北京", confidence=0.5)
    b1 = _fact(fact_id="f-3", subject="Lisa", predicate="喜欢", object_="猫", confidence=0.8)
    semantic.upsert_fact(a2)

    resolved = resolve_conflicts([a1, a2, b1], semantic=semantic)

    assert {fact.id for fact in resolved} == {"f-1", "f-3"}
    assert semantic.get_fact("f-2").status == "superseded"  # type: ignore[union-attr]
