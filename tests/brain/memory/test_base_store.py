"""BaseStore（`base` 层：原始对话单元）的会话/单元/链接回归测试。"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from miku_on_desk.brain.memory.base_store import BaseStore
from miku_on_desk.brain.memory.models import MemoryUnit


@pytest.fixture
def store(tmp_path: Path) -> BaseStore:
    return BaseStore(tmp_path / "base", index_path=tmp_path / "index.json")


def _unit(
    *,
    unit_id: str = "",
    session_id: str = "s1",
    role: str = "user",
    content: str = "你好",
    created_at: str = "2026-07-06T09:00:00+00:00",
) -> MemoryUnit:
    return MemoryUnit(
        id=unit_id,
        session_id=session_id,
        role=role,  # type: ignore[arg-type]
        content=content,
        created_at=created_at,
    )


# ── append / load ────────────────────────────────────────────────────────


def test_append_generates_id_when_missing_and_load_roundtrips(store: BaseStore) -> None:
    unit_id = store.append(_unit())

    assert unit_id
    loaded = store.load(unit_id)
    assert loaded is not None
    assert loaded.id == unit_id
    assert loaded.session_id == "s1"
    assert loaded.role == "user"
    assert loaded.content == "你好"
    assert loaded.created_at == "2026-07-06T09:00:00+00:00"


def test_append_preserves_explicit_id(store: BaseStore) -> None:
    unit_id = store.append(_unit(unit_id="fixed-id"))

    assert unit_id == "fixed-id"
    assert store.load("fixed-id") is not None


def test_append_writes_file_under_date_directory(store: BaseStore, tmp_path: Path) -> None:
    unit_id = store.append(_unit(created_at="2026-07-06T09:00:00+00:00"))

    expected_path = tmp_path / "base" / "2026-07-06" / f"u_{unit_id}.md"
    assert expected_path.exists()


def test_load_returns_none_for_unknown_id(store: BaseStore) -> None:
    assert store.load("does-not-exist") is None


# ── list_units ───────────────────────────────────────────────────────────


def test_list_units_returns_session_units_in_chronological_order(store: BaseStore) -> None:
    store.append(_unit(unit_id="a", created_at="2026-07-06T09:00:00+00:00", content="第一条"))
    store.append(_unit(unit_id="b", created_at="2026-07-06T09:05:00+00:00", content="第二条"))

    units = store.list_units(session_id="s1")

    assert [unit.id for unit in units] == ["a", "b"]


def test_list_units_filters_by_session(store: BaseStore) -> None:
    store.append(_unit(unit_id="a", session_id="s1"))
    store.append(_unit(unit_id="b", session_id="s2"))

    units = store.list_units(session_id="s1")

    assert [unit.id for unit in units] == ["a"]


def test_list_units_limit_returns_most_recent(store: BaseStore) -> None:
    store.append(_unit(unit_id="a", created_at="2026-07-06T09:00:00+00:00"))
    store.append(_unit(unit_id="b", created_at="2026-07-06T09:05:00+00:00"))
    store.append(_unit(unit_id="c", created_at="2026-07-06T09:10:00+00:00"))

    units = store.list_units(session_id="s1", limit=2)

    assert [unit.id for unit in units] == ["b", "c"]


# ── search ───────────────────────────────────────────────────────────────


def test_search_matches_substring_case_insensitively(store: BaseStore) -> None:
    store.append(_unit(unit_id="a", content="今天天气不错"))
    store.append(_unit(unit_id="b", content="Hello World"))

    hits = store.search("hello")

    assert [unit.id for unit in hits] == ["b"]


def test_search_scopes_to_session_when_given(store: BaseStore) -> None:
    store.append(_unit(unit_id="a", session_id="s1", content="共同关键词"))
    store.append(_unit(unit_id="b", session_id="s2", content="共同关键词"))

    hits = store.search("关键词", session_id="s1")

    assert [unit.id for unit in hits] == ["a"]


def test_search_returns_empty_for_blank_query(store: BaseStore) -> None:
    store.append(_unit())

    assert store.search("   ") == []


# ── session 注册表 ────────────────────────────────────────────────────────


def test_start_session_then_list_sessions_orders_by_most_recently_updated(
    store: BaseStore,
) -> None:
    store.start_session("s1", "旧会话")
    store.start_session("s2", "新会话")
    store.append(_unit(unit_id="a", session_id="s1", created_at="2099-01-01T00:00:00+00:00"))

    sessions = store.list_sessions()

    assert [meta.session_id for meta in sessions] == ["s1", "s2"]


def test_update_session_title_changes_title(store: BaseStore) -> None:
    store.start_session("s1", "旧标题")

    store.update_session_title("s1", "新标题")

    sessions = store.list_sessions()
    assert sessions[0].title == "新标题"


def test_update_session_title_is_noop_for_missing_session(store: BaseStore) -> None:
    store.update_session_title("does-not-exist", "新标题")

    assert store.list_sessions() == []


def test_delete_session_removes_from_registry_but_keeps_unit_file_on_disk(
    store: BaseStore, tmp_path: Path
) -> None:
    store.start_session("s1", "会话")
    unit_id = store.append(_unit(unit_id="a", session_id="s1"))

    store.delete_session("s1")

    assert store.list_sessions() == []
    assert store.list_units(session_id="s1") == []
    assert store.load(unit_id) is not None


# ── 链接 ─────────────────────────────────────────────────────────────────


def test_link_temporal_appends_jsonl_line(store: BaseStore, tmp_path: Path) -> None:
    store.link_temporal("a", "b")

    links_path = tmp_path / "base" / "_links.jsonl"
    assert links_path.exists()
    assert '"from": "a"' in links_path.read_text(encoding="utf-8")


def test_link_temporal_noop_when_prev_id_is_none(store: BaseStore, tmp_path: Path) -> None:
    store.link_temporal(None, "b")

    links_path = tmp_path / "base" / "_links.jsonl"
    assert not links_path.exists()


def test_find_semantically_similar_returns_matches_above_threshold(store: BaseStore) -> None:
    store.append(_unit(unit_id="a", content="今天天气真好适合出门散步"))
    store.append(_unit(unit_id="b", content="今天天气真好适合出门散步呀"))
    store.append(_unit(unit_id="c", content="量子计算机的原理是什么"))
    query = _unit(unit_id="q", content="今天天气真好适合出门散步")

    similar = store.find_semantically_similar(query, threshold=0.5)

    similar_ids = [unit_id for unit_id, _score in similar]
    assert "a" in similar_ids
    assert "b" in similar_ids
    assert "c" not in similar_ids


def test_find_semantically_similar_with_session_id_restricts_scope_to_session(
    store: BaseStore,
) -> None:
    store.append(_unit(unit_id="a", session_id="s1", content="今天天气真好适合出门散步"))
    store.append(_unit(unit_id="b", session_id="s2", content="今天天气真好适合出门散步呀"))
    query = _unit(unit_id="q", session_id="s1", content="今天天气真好适合出门散步")

    similar = store.find_semantically_similar(query, threshold=0.5, session_id="s1")

    similar_ids = [unit_id for unit_id, _score in similar]
    assert similar_ids == ["a"]


def test_find_semantically_similar_without_session_id_still_scans_all_sessions(
    store: BaseStore,
) -> None:
    store.append(_unit(unit_id="a", session_id="s1", content="今天天气真好适合出门散步"))
    store.append(_unit(unit_id="b", session_id="s2", content="今天天气真好适合出门散步呀"))
    query = _unit(unit_id="q", session_id="s1", content="今天天气真好适合出门散步")

    similar = store.find_semantically_similar(query, threshold=0.5)

    similar_ids = [unit_id for unit_id, _score in similar]
    assert "a" in similar_ids
    assert "b" in similar_ids


# ── consolidation 时间戳 ──────────────────────────────────────────────────


def test_last_consolidated_defaults_to_none_and_roundtrips(store: BaseStore) -> None:
    assert store.get_last_consolidated() is None

    store.set_last_consolidated("2026-07-06T12:00:00+00:00")

    assert store.get_last_consolidated() == "2026-07-06T12:00:00+00:00"


# ── 并发写保护（阶段 E） ──────────────────────────────────────────────────


def test_concurrent_append_from_two_threads_has_no_lost_updates(store: BaseStore) -> None:
    """两个线程同时对同一个 `BaseStore` 写 `index.json`，`RLock` 应避免"读改写"竞态丢更新。"""
    errors: list[BaseException] = []

    def _append_many(prefix: str) -> None:
        try:
            for i in range(20):
                store.append(_unit(unit_id=f"{prefix}{i}", content=f"消息{i}"))
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=_append_many, args=("a",)),
        threading.Thread(target=_append_many, args=("b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    units = store.list_units()
    assert len(units) == 40
    assert len({unit.id for unit in units}) == 40
