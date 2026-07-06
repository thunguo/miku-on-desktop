"""SemanticStore（`semantic` 层：事实三元组 + 实体）的 CRUD/查询回归测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from miku_on_desk.brain.memory.models import Entity, Fact
from miku_on_desk.brain.memory.semantic_store import SemanticStore


@pytest.fixture
def store(tmp_path: Path) -> SemanticStore:
    return SemanticStore(tmp_path / "semantic")


def _fact(
    *,
    fact_id: str = "",
    subject: str = "用户",
    predicate: str = "喜欢",
    object_: str = "猫",
    confidence: float = 0.9,
    status: str = "active",
    pinned: bool = False,
    recorded_at: str = "2026-07-06T09:00:00+00:00",
    extracted_by: str = "llm:fast",
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
        valid_from=recorded_at,
        recorded_at=recorded_at,
        extracted_by=extracted_by,
        status=status,  # type: ignore[arg-type]
        pinned=pinned,
    )


def _entity(
    *,
    entity_id: str = "",
    name: str = "Lisa",
    aliases: list[str] | None = None,
    mention_count: int = 1,
) -> Entity:
    return Entity(
        id=entity_id,
        name=name,
        type="person",
        aliases=aliases or [],
        first_seen="2026-07-06T09:00:00+00:00",
        last_mentioned="2026-07-06T09:00:00+00:00",
        mention_count=mention_count,
    )


# ── facts ────────────────────────────────────────────────────────────────


def test_upsert_fact_generates_id_when_missing_and_roundtrips(store: SemanticStore) -> None:
    fact_id = store.upsert_fact(_fact())

    assert fact_id
    loaded = store.get_fact(fact_id)
    assert loaded is not None
    assert loaded.subject == "用户"
    assert loaded.predicate == "喜欢"
    assert loaded.object == "猫"


def test_upsert_fact_with_existing_id_replaces_in_place(store: SemanticStore) -> None:
    fact_id = store.upsert_fact(_fact())
    store.upsert_fact(_fact(fact_id=fact_id, object_="狗"))

    assert store.list_facts() == [store.get_fact(fact_id)]
    assert store.get_fact(fact_id).object == "狗"  # type: ignore[union-attr]


def test_get_fact_returns_none_for_unknown_id(store: SemanticStore) -> None:
    assert store.get_fact("does-not-exist") is None


def test_list_facts_filters_by_subject_and_status(store: SemanticStore) -> None:
    store.upsert_fact(_fact(subject="用户", status="active"))
    store.upsert_fact(_fact(subject="Lisa", status="superseded"))

    assert [f.subject for f in store.list_facts(subject="用户")] == ["用户"]
    assert [f.status for f in store.list_facts(status="superseded")] == ["superseded"]


def test_delete_fact_removes_it(store: SemanticStore) -> None:
    fact_id = store.upsert_fact(_fact())

    store.delete_fact(fact_id)

    assert store.get_fact(fact_id) is None


def test_list_pinned_facts_returns_only_active_and_pinned(store: SemanticStore) -> None:
    store.upsert_fact(_fact(subject="a", pinned=True, status="active"))
    store.upsert_fact(_fact(subject="b", pinned=True, status="superseded"))
    store.upsert_fact(_fact(subject="c", pinned=False, status="active"))

    pinned = store.list_pinned_facts()

    assert [f.subject for f in pinned] == ["a"]


def test_search_facts_matches_subject_predicate_object_and_context(store: SemanticStore) -> None:
    store.upsert_fact(_fact(subject="用户", predicate="住在", object_="上海"))
    store.upsert_fact(_fact(subject="Lisa", predicate="工作于", object_="初创公司"))

    hits = store.search_facts("上海")

    assert [f.object for f in hits] == ["上海"]


def test_search_facts_returns_empty_for_blank_query(store: SemanticStore) -> None:
    store.upsert_fact(_fact())

    assert store.search_facts("   ") == []


def test_facts_persist_across_store_instances(store: SemanticStore, tmp_path: Path) -> None:
    fact_id = store.upsert_fact(_fact())

    reopened = SemanticStore(tmp_path / "semantic")

    assert reopened.get_fact(fact_id) is not None


# ── entities ─────────────────────────────────────────────────────────────


def test_upsert_entity_generates_id_when_missing_and_roundtrips(store: SemanticStore) -> None:
    entity_id = store.upsert_entity(_entity())

    assert entity_id
    loaded = store.get_entity(entity_id)
    assert loaded is not None
    assert loaded.name == "Lisa"


def test_upsert_entity_with_existing_id_replaces_in_place(store: SemanticStore) -> None:
    entity_id = store.upsert_entity(_entity())
    store.upsert_entity(_entity(entity_id=entity_id, mention_count=5))

    assert store.list_entities() == [store.get_entity(entity_id)]
    assert store.get_entity(entity_id).mention_count == 5  # type: ignore[union-attr]


def test_find_entity_by_name_matches_name_case_insensitively(store: SemanticStore) -> None:
    store.upsert_entity(_entity(name="Lisa"))

    found = store.find_entity_by_name("lisa")

    assert found is not None
    assert found.name == "Lisa"


def test_find_entity_by_name_matches_alias(store: SemanticStore) -> None:
    store.upsert_entity(_entity(name="Lisa", aliases=["丽莎"]))

    found = store.find_entity_by_name("丽莎")

    assert found is not None
    assert found.name == "Lisa"


def test_find_entity_by_name_returns_none_when_no_match(store: SemanticStore) -> None:
    assert store.find_entity_by_name("不存在") is None


def test_delete_entity_removes_it(store: SemanticStore) -> None:
    entity_id = store.upsert_entity(_entity())

    store.delete_entity(entity_id)

    assert store.get_entity(entity_id) is None


def test_touch_entity_mention_bumps_count_and_last_mentioned(store: SemanticStore) -> None:
    entity_id = store.upsert_entity(_entity(mention_count=1))

    store.touch_entity_mention(entity_id, mentioned_at="2026-07-06T12:00:00+00:00")

    updated = store.get_entity(entity_id)
    assert updated is not None
    assert updated.mention_count == 2
    assert updated.last_mentioned == "2026-07-06T12:00:00+00:00"


def test_touch_entity_mention_is_noop_for_unknown_id(store: SemanticStore) -> None:
    store.touch_entity_mention("does-not-exist", mentioned_at="2026-07-06T12:00:00+00:00")

    assert store.list_entities() == []
