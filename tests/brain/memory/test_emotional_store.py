"""EmotionalStore（`emotional` 层：偏好 + 信任模型 JSON）的读写回归测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from miku_on_desk.brain.memory.emotional_store import EmotionalStore


@pytest.fixture
def store(tmp_path: Path) -> EmotionalStore:
    return EmotionalStore(tmp_path / "emotional")


# ── preferences ──────────────────────────────────────────────────────────


def test_load_preferences_returns_default_skeleton_when_missing(store: EmotionalStore) -> None:
    preferences = store.load_preferences()

    assert preferences["version"] == "1.0"
    assert preferences["confidence_threshold"] == 0.75


def test_save_preferences_then_load_roundtrips_nested_structure(store: EmotionalStore) -> None:
    data = {
        "version": "1.0",
        "last_updated": "2026-07-06T10:00:00Z",
        "confidence_threshold": 0.75,
        "location_preferences": {
            "familiar_cities": [
                {
                    "city": "上海",
                    "confidence": 0.95,
                    "source_facts": ["f-002", "f-004"],
                }
            ]
        },
    }

    store.save_preferences(data)

    loaded = store.load_preferences()
    assert loaded == data


def test_save_preferences_overwrites_previous_content(store: EmotionalStore) -> None:
    store.save_preferences({"version": "1.0", "note": "第一次"})
    store.save_preferences({"version": "1.0", "note": "第二次"})

    assert store.load_preferences()["note"] == "第二次"


def test_preferences_persist_across_store_instances(store: EmotionalStore, tmp_path: Path) -> None:
    store.save_preferences({"version": "1.0", "note": "持久化"})

    reopened = EmotionalStore(tmp_path / "emotional")

    assert reopened.load_preferences()["note"] == "持久化"


def test_preferences_file_written_under_root(store: EmotionalStore, tmp_path: Path) -> None:
    store.save_preferences({"version": "1.0"})

    assert (tmp_path / "emotional" / "preferences.json").exists()


# ── trust model ──────────────────────────────────────────────────────────


def test_load_trust_model_returns_default_skeleton_when_missing(store: EmotionalStore) -> None:
    trust_model = store.load_trust_model()

    assert trust_model["version"] == "1.0"
    assert trust_model["decay_model"]["half_life_days"] == 30
    assert trust_model["fact_trust_scores"] == {}


def test_save_trust_model_then_load_roundtrips(store: EmotionalStore) -> None:
    data = {
        "version": "1.0",
        "last_updated": "2026-07-06T10:00:00Z",
        "fact_trust_scores": {
            "f-001": {"score": 0.95, "basis": "用户直接陈述", "verification_count": 1}
        },
        "entity_consistency": {},
        "decay_model": {
            "half_life_days": 30,
            "last_access_boost": 0.1,
            "repeated_confirmation_boost": 0.05,
        },
    }

    store.save_trust_model(data)

    assert store.load_trust_model() == data


def test_trust_model_persists_across_store_instances(
    store: EmotionalStore, tmp_path: Path
) -> None:
    store.save_trust_model({"version": "1.0", "fact_trust_scores": {"f-001": {"score": 0.9}}})

    reopened = EmotionalStore(tmp_path / "emotional")

    assert reopened.load_trust_model()["fact_trust_scores"]["f-001"]["score"] == 0.9


def test_preferences_and_trust_model_are_independent_files(
    store: EmotionalStore, tmp_path: Path
) -> None:
    store.save_preferences({"version": "1.0", "note": "偏好"})
    store.save_trust_model({"version": "1.0", "note": "信任"})

    assert store.load_preferences()["note"] == "偏好"
    assert store.load_trust_model()["note"] == "信任"
    assert (tmp_path / "emotional" / "preferences.json").exists()
    assert (tmp_path / "emotional" / "trust_model.json").exists()
