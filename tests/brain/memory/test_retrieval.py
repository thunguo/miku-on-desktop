"""retrieval.py（跨格式混合检索）的回归测试：三路搜索 + token 预算组装 + 轻量提示两条入口。"""

from __future__ import annotations

from pathlib import Path

import pytest

from miku_on_desk.brain.memory.emotional_store import EmotionalStore
from miku_on_desk.brain.memory.episodic_store import EpisodicStore
from miku_on_desk.brain.memory.models import Fact
from miku_on_desk.brain.memory.retrieval import retrieve, retrieve_hints
from miku_on_desk.brain.memory.semantic_store import SemanticStore


@pytest.fixture
def semantic(tmp_path: Path) -> SemanticStore:
    return SemanticStore(tmp_path / "semantic")


@pytest.fixture
def episodic(tmp_path: Path) -> EpisodicStore:
    return EpisodicStore(tmp_path / "episodic")


@pytest.fixture
def emotional(tmp_path: Path) -> EmotionalStore:
    return EmotionalStore(tmp_path / "emotional")


def _fact(
    *,
    subject: str = "用户",
    predicate: str = "住在",
    object_: str = "上海",
    confidence: float = 0.9,
    status: str = "active",
    recorded_at: str = "2026-07-06T09:00:00+00:00",
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
        valid_from=recorded_at,
        recorded_at=recorded_at,
        extracted_by="llm:fast",
        status=status,  # type: ignore[arg-type]
    )


# ── retrieve_hints ───────────────────────────────────────────────────────


def test_retrieve_hints_includes_active_semantic_fact(
    semantic: SemanticStore, episodic: EpisodicStore, emotional: EmotionalStore
) -> None:
    semantic.upsert_fact(_fact(object_="上海"))

    hints = retrieve_hints(semantic=semantic, episodic=episodic, emotional=emotional, query="上海")

    assert any(hint.label == "语义" and "上海" in hint.text for hint in hints)


def test_retrieve_hints_excludes_superseded_semantic_fact(
    semantic: SemanticStore, episodic: EpisodicStore, emotional: EmotionalStore
) -> None:
    semantic.upsert_fact(_fact(object_="上海", status="superseded"))

    hints = retrieve_hints(semantic=semantic, episodic=episodic, emotional=emotional, query="上海")

    assert hints == []


def test_retrieve_hints_includes_matching_episode(
    semantic: SemanticStore, episodic: EpisodicStore, emotional: EmotionalStore
) -> None:
    episodic.append_event(
        title="搬家",
        summary="用户提到搬到了北京。",
        occurred_at="2026-07-06T09:00:00+00:00",
    )

    hints = retrieve_hints(semantic=semantic, episodic=episodic, emotional=emotional, query="搬家")

    assert any(hint.label == "情景" and "搬家" in hint.text for hint in hints)


def test_retrieve_hints_includes_matching_emotional_preference(
    semantic: SemanticStore, episodic: EpisodicStore, emotional: EmotionalStore
) -> None:
    emotional.save_preferences({"location_preferences": {"familiar_cities": ["上海"]}})

    hints = retrieve_hints(semantic=semantic, episodic=episodic, emotional=emotional, query="上海")

    assert any(hint.label == "情感" for hint in hints)


def test_retrieve_hints_truncates_to_limit_across_channels(
    semantic: SemanticStore, episodic: EpisodicStore, emotional: EmotionalStore
) -> None:
    for i in range(5):
        semantic.upsert_fact(_fact(predicate=f"喜欢{i}", object_="猫"))

    hints = retrieve_hints(
        semantic=semantic, episodic=episodic, emotional=emotional, query="猫", limit=2
    )

    assert len(hints) == 2


def test_retrieve_hints_empty_query_yields_no_hints(
    semantic: SemanticStore, episodic: EpisodicStore, emotional: EmotionalStore
) -> None:
    semantic.upsert_fact(_fact())
    episodic.append_event(
        title="标题", summary="摘要", occurred_at="2026-07-06T09:00:00+00:00"
    )
    emotional.save_preferences({"a": "b"})

    hints = retrieve_hints(semantic=semantic, episodic=episodic, emotional=emotional, query="")

    assert hints == []


# ── retrieve ─────────────────────────────────────────────────────────────


def test_retrieve_assembles_all_three_sections(
    semantic: SemanticStore, episodic: EpisodicStore, emotional: EmotionalStore
) -> None:
    semantic.upsert_fact(_fact(object_="上海", confidence=0.9))
    episodic.append_event(
        title="搬家到上海",
        summary="用户提到搬到了上海。",
        occurred_at="2026-07-06T09:00:00+00:00",
    )
    emotional.save_preferences({"location_preferences": {"familiar_cities": ["上海"]}})

    text = retrieve(semantic=semantic, episodic=episodic, emotional=emotional, query="上海")

    assert "已知事实" in text
    assert "相关事件" in text
    assert "偏好参考" in text


def test_retrieve_excludes_facts_below_confidence_threshold(
    semantic: SemanticStore, episodic: EpisodicStore, emotional: EmotionalStore
) -> None:
    semantic.upsert_fact(_fact(object_="上海", confidence=0.5))

    text = retrieve(semantic=semantic, episodic=episodic, emotional=emotional, query="上海")

    assert "已知事实" not in text


def test_retrieve_excludes_superseded_facts(
    semantic: SemanticStore, episodic: EpisodicStore, emotional: EmotionalStore
) -> None:
    semantic.upsert_fact(_fact(object_="上海", confidence=0.9, status="superseded"))

    text = retrieve(semantic=semantic, episodic=episodic, emotional=emotional, query="上海")

    assert "已知事实" not in text


def test_retrieve_sorts_semantic_facts_by_confidence_descending(
    semantic: SemanticStore, episodic: EpisodicStore, emotional: EmotionalStore
) -> None:
    semantic.upsert_fact(_fact(predicate="喜欢", object_="猫", confidence=0.75))
    semantic.upsert_fact(_fact(predicate="喜欢", object_="狗", confidence=0.95))

    text = retrieve(semantic=semantic, episodic=episodic, emotional=emotional, query="喜欢")

    dog_index = text.index("狗")
    cat_index = text.index("猫")
    assert dog_index < cat_index


def test_retrieve_returns_empty_string_when_nothing_matches(
    semantic: SemanticStore, episodic: EpisodicStore, emotional: EmotionalStore
) -> None:
    text = retrieve(semantic=semantic, episodic=episodic, emotional=emotional, query="不存在的东西")

    assert text == ""


def test_retrieve_truncates_semantic_section_to_token_budget(
    semantic: SemanticStore, episodic: EpisodicStore, emotional: EmotionalStore
) -> None:
    for i in range(50):
        semantic.upsert_fact(
            _fact(predicate="喜欢", object_=f"事物{i:03d}" * 10, confidence=0.9)
        )

    text = retrieve(
        semantic=semantic, episodic=episodic, emotional=emotional, query="喜欢", token_budget=100
    )

    semantic_section = next(
        section for section in text.split("\n\n") if section.startswith("已知事实")
    )
    line_count = semantic_section.count("\n")
    assert line_count < 50
