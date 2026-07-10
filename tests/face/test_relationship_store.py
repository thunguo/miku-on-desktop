"""character_relationships.json 读写往返、损坏文件回退、bump 递增持久化。"""

from __future__ import annotations

import json
from pathlib import Path

from miku_on_desk.face.relationship_store import RelationshipStore


def test_relationship_store_load_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    store = RelationshipStore(tmp_path / "character_relationships.json")

    assert store.load() == {}


def test_relationship_store_save_then_load_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "character_relationships.json"
    store = RelationshipStore(path)

    store.save({"miku_pixel": 3, "tew": 1})

    assert store.load() == {"miku_pixel": 3, "tew": 1}


def test_relationship_store_load_recovers_from_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "character_relationships.json"
    path.write_text("not json", encoding="utf-8")
    store = RelationshipStore(path)

    assert store.load() == {}


def test_relationship_store_save_overwrites_without_leaving_tmp_files(tmp_path: Path) -> None:
    path = tmp_path / "character_relationships.json"
    store = RelationshipStore(path)

    store.save({"miku_pixel": 1})
    store.save({"miku_pixel": 2})

    assert json.loads(path.read_text(encoding="utf-8"))["familiarity"]["miku_pixel"] == 2
    assert list(path.parent.glob("*.tmp-*")) == []


def test_relationship_store_get_returns_zero_when_unknown(tmp_path: Path) -> None:
    store = RelationshipStore(tmp_path / "character_relationships.json")

    assert store.get("miku_pixel") == 0


def test_relationship_store_bump_increments_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "character_relationships.json"
    store = RelationshipStore(path)

    first = store.bump("miku_pixel")
    second = store.bump("miku_pixel")

    assert first == 1
    assert second == 2
    assert RelationshipStore(path).get("miku_pixel") == 2


def test_relationship_store_bump_tracks_separate_characters_independently(
    tmp_path: Path,
) -> None:
    store = RelationshipStore(tmp_path / "character_relationships.json")

    store.bump("miku_pixel")
    store.bump("tew")
    store.bump("miku_pixel")

    assert store.get("miku_pixel") == 2
    assert store.get("tew") == 1
