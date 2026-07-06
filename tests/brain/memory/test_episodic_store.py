"""EpisodicStore（`episodic` 层：按月 Markdown 事件链）的 CRUD/查询回归测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from miku_on_desk.brain.memory.episodic_store import EpisodicStore


@pytest.fixture
def store(tmp_path: Path) -> EpisodicStore:
    return EpisodicStore(tmp_path / "episodic")


def _append(
    store: EpisodicStore,
    *,
    title: str = "第一次见面",
    summary: str = "用户提到自己住在上海静安区，在一家初创公司工作。",
    occurred_at: str = "2026-07-06T09:00:00+00:00",
    source_units: list[str] | None = None,
    emotion_tag: str | None = None,
    participants: list[str] | None = None,
    event_chain: list[str] | None = None,
    related_events: list[str] | None = None,
    session_id: str | None = None,
    model: str | None = None,
) -> str:
    return store.append_event(
        title=title,
        summary=summary,
        occurred_at=occurred_at,
        source_units=source_units,
        emotion_tag=emotion_tag,
        participants=participants,
        event_chain=event_chain,
        related_events=related_events,
        session_id=session_id,
        model=model,
    )


# ── append / get ─────────────────────────────────────────────────────────


def test_append_event_generates_sequential_ids(store: EpisodicStore) -> None:
    first_id = _append(store)
    second_id = _append(store)

    assert first_id == "E:001"
    assert second_id == "E:002"


def test_append_event_roundtrips_all_fields(store: EpisodicStore) -> None:
    event_id = _append(
        store,
        title="Lisa 的近况",
        summary="Lisa 搬到了上海。",
        occurred_at="2026-07-06T09:00:00+00:00",
        source_units=["u_001", "u_002"],
        emotion_tag="温暖",
        participants=["Lisa"],
        event_chain=["提到搬家", "确认新地址"],
        related_events=["E:000"],
        session_id="s_001",
        model="claude-sonnet-4-6",
    )

    episode = store.get_event(event_id)

    assert episode is not None
    assert episode.title == "Lisa 的近况"
    assert episode.summary == "Lisa 搬到了上海。"
    assert episode.occurred_at == "2026-07-06T09:00:00+00:00"
    assert episode.source_units == ["u_001", "u_002"]
    assert episode.emotion_tag == "温暖"
    assert episode.participants == ["Lisa"]
    assert episode.event_chain == ["提到搬家", "确认新地址"]
    assert episode.related_events == ["E:000"]
    assert episode.session_id == "s_001"
    assert episode.model == "claude-sonnet-4-6"


def test_append_event_creates_month_file_under_year_directory(
    store: EpisodicStore, tmp_path: Path
) -> None:
    _append(store, occurred_at="2026-07-06T09:00:00+00:00")

    assert (tmp_path / "episodic" / "2026" / "2026-07.md").exists()


def test_get_event_returns_none_for_unknown_id(store: EpisodicStore) -> None:
    assert store.get_event("E:999") is None


def test_append_event_without_optional_fields_omits_them_on_roundtrip(
    store: EpisodicStore,
) -> None:
    event_id = _append(store, emotion_tag=None, participants=None, session_id=None, model=None)

    episode = store.get_event(event_id)

    assert episode is not None
    assert episode.emotion_tag is None
    assert episode.participants == []
    assert episode.session_id is None
    assert episode.model is None


# ── list_events / list_months ───────────────────────────────────────────


def test_list_events_orders_chronologically_across_months(store: EpisodicStore) -> None:
    _append(store, title="七月", occurred_at="2026-07-06T09:00:00+00:00")
    _append(store, title="一月", occurred_at="2026-01-06T09:00:00+00:00")

    titles = [episode.title for episode in store.list_events()]

    assert titles == ["一月", "七月"]


def test_list_events_filters_by_month(store: EpisodicStore) -> None:
    _append(store, title="七月事件", occurred_at="2026-07-06T09:00:00+00:00")
    _append(store, title="一月事件", occurred_at="2026-01-06T09:00:00+00:00")

    titles = [episode.title for episode in store.list_events(month="2026-07")]

    assert titles == ["七月事件"]


def test_list_events_limit_returns_most_recent(store: EpisodicStore) -> None:
    _append(store, title="第一", occurred_at="2026-01-01T09:00:00+00:00")
    _append(store, title="第二", occurred_at="2026-02-01T09:00:00+00:00")
    _append(store, title="第三", occurred_at="2026-03-01T09:00:00+00:00")

    titles = [episode.title for episode in store.list_events(limit=2)]

    assert titles == ["第二", "第三"]


def test_list_months_returns_sorted_distinct_months(store: EpisodicStore) -> None:
    _append(store, occurred_at="2026-07-06T09:00:00+00:00")
    _append(store, occurred_at="2026-01-06T09:00:00+00:00")
    _append(store, occurred_at="2026-07-20T09:00:00+00:00")

    assert store.list_months() == ["2026-01", "2026-07"]


# ── search ───────────────────────────────────────────────────────────────


def test_search_matches_title_case_insensitively(store: EpisodicStore) -> None:
    _append(store, title="Lisa 搬家了")

    hits = store.search("lisa")

    assert [episode.title for episode in hits] == ["Lisa 搬家了"]


def test_search_matches_summary(store: EpisodicStore) -> None:
    _append(store, title="第一次见面", summary="提到住在静安区。")

    hits = store.search("静安区")

    assert len(hits) == 1


def test_search_matches_participants(store: EpisodicStore) -> None:
    _append(store, title="事件", participants=["Lisa"])

    hits = store.search("lisa")

    assert len(hits) == 1


def test_search_matches_event_chain(store: EpisodicStore) -> None:
    _append(store, title="事件", event_chain=["提到搬家到静安区"])

    hits = store.search("静安区")

    assert len(hits) == 1


def test_search_returns_empty_for_blank_query(store: EpisodicStore) -> None:
    _append(store)

    assert store.search("   ") == []


# ── update_summary / delete_event ───────────────────────────────────────


def test_update_summary_replaces_summary_only(store: EpisodicStore) -> None:
    event_id = _append(store, title="标题不变", summary="旧摘要")

    store.update_summary(event_id, "新摘要")

    episode = store.get_event(event_id)
    assert episode is not None
    assert episode.title == "标题不变"
    assert episode.summary == "新摘要"


def test_update_summary_is_noop_for_unknown_id(store: EpisodicStore) -> None:
    store.update_summary("E:999", "新摘要")


def test_delete_event_removes_it(store: EpisodicStore) -> None:
    event_id = _append(store)

    store.delete_event(event_id)

    assert store.get_event(event_id) is None


def test_delete_event_is_noop_for_unknown_id(store: EpisodicStore) -> None:
    store.delete_event("E:999")


# ── _index.md ────────────────────────────────────────────────────────────


def test_append_event_rebuilds_index_with_entry(store: EpisodicStore, tmp_path: Path) -> None:
    event_id = _append(store, title="第一次见面")

    index_text = (tmp_path / "episodic" / "_index.md").read_text(encoding="utf-8")

    assert event_id in index_text
    assert "第一次见面" in index_text


def test_delete_event_removes_entry_from_index(store: EpisodicStore, tmp_path: Path) -> None:
    event_id = _append(store, title="待删除事件")

    store.delete_event(event_id)

    index_text = (tmp_path / "episodic" / "_index.md").read_text(encoding="utf-8")
    assert "待删除事件" not in index_text


# ── persistence across instances ────────────────────────────────────────


def test_events_persist_across_store_instances(store: EpisodicStore, tmp_path: Path) -> None:
    event_id = _append(store)

    reopened = EpisodicStore(tmp_path / "episodic")

    assert reopened.get_event(event_id) is not None


def test_next_event_id_continues_after_reopen(store: EpisodicStore, tmp_path: Path) -> None:
    _append(store)

    reopened = EpisodicStore(tmp_path / "episodic")
    second_id = _append(reopened)

    assert second_id == "E:002"
