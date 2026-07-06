"""`semantic` 层：JSONL 存储的事实三元组与实体定义。

只做机械的 CRUD/查询，不做去重合并——「同一实体是否已存在」「新事实是否与旧事实冲突」这类
决策属于 `conflict.py`/`extraction.py` 编排层的职责（参见计划文档），这里保持和
`base_store.py` 一致的"哑"存储层定位，方便独立测试。

设计文档 §4.2 的 `edges.jsonl`/`inferred.jsonl` 没有单独建文件：实体关系可以从
`facts.jsonl` 里 subject/object 两端解析，"推断 vs 明确陈述"的区分用 `confidence`/
`extracted_by` 字段表达，不额外维护一份物化的边表。
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from miku_on_desk.brain.memory.models import Entity, EntityType, Fact, FactStatus

_VALID_ENTITY_TYPES: frozenset[str] = frozenset(
    {"person", "location", "organization", "concept", "event", "technology"}
)
_VALID_FACT_STATUSES: frozenset[str] = frozenset(
    {"active", "superseded", "conflict", "archived"}
)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp-{uuid.uuid4().hex}")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _entity_type_from_str(value: str) -> EntityType:
    if value not in _VALID_ENTITY_TYPES:
        raise ValueError(f"未知的实体类型：{value!r}")
    return cast(EntityType, value)


def _fact_status_from_str(value: str) -> FactStatus:
    if value not in _VALID_FACT_STATUSES:
        raise ValueError(f"未知的事实状态：{value!r}")
    return cast(FactStatus, value)


def _fact_from_dict(data: dict[str, Any]) -> Fact:
    return Fact(
        id=data["id"],
        subject=data["subject"],
        subject_type=_entity_type_from_str(data["subject_type"]),
        predicate=data["predicate"],
        object=data["object"],
        object_type=_entity_type_from_str(data["object_type"]),
        confidence=float(data["confidence"]),
        source=list(data.get("source", [])),
        valid_from=data["valid_from"],
        recorded_at=data["recorded_at"],
        extracted_by=data["extracted_by"],
        status=_fact_status_from_str(data["status"]),
        context=data.get("context"),
        pinned=bool(data.get("pinned", False)),
    )


def _fact_to_dict(fact: Fact) -> dict[str, Any]:
    return {
        "id": fact.id,
        "subject": fact.subject,
        "subject_type": fact.subject_type,
        "predicate": fact.predicate,
        "object": fact.object,
        "object_type": fact.object_type,
        "confidence": fact.confidence,
        "source": fact.source,
        "valid_from": fact.valid_from,
        "recorded_at": fact.recorded_at,
        "extracted_by": fact.extracted_by,
        "status": fact.status,
        "context": fact.context,
        "pinned": fact.pinned,
    }


def _entity_from_dict(data: dict[str, Any]) -> Entity:
    return Entity(
        id=data["id"],
        name=data["name"],
        type=_entity_type_from_str(data["type"]),
        aliases=list(data.get("aliases", [])),
        first_seen=data.get("first_seen", ""),
        last_mentioned=data.get("last_mentioned", ""),
        mention_count=int(data.get("mention_count", 0)),
    )


def _entity_to_dict(entity: Entity) -> dict[str, Any]:
    return {
        "id": entity.id,
        "name": entity.name,
        "type": entity.type,
        "aliases": entity.aliases,
        "first_seen": entity.first_seen,
        "last_mentioned": entity.last_mentioned,
        "mention_count": entity.mention_count,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [cast(dict[str, Any], json.loads(line)) for line in lines if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    _atomic_write_text(path, content + "\n" if rows else "")


class SemanticStore:
    """`semantic` 层存储：`facts.jsonl` + `entities.jsonl`。"""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._facts_path = root / "facts.jsonl"
        self._entities_path = root / "entities.jsonl"
        self._root.mkdir(parents=True, exist_ok=True)

    # ── facts ────────────────────────────────────────────────────────────

    def list_facts(
        self, *, subject: str | None = None, status: FactStatus | None = None
    ) -> list[Fact]:
        facts = [_fact_from_dict(row) for row in _read_jsonl(self._facts_path)]
        if subject is not None:
            facts = [fact for fact in facts if fact.subject == subject]
        if status is not None:
            facts = [fact for fact in facts if fact.status == status]
        return facts

    def get_fact(self, fact_id: str) -> Fact | None:
        for fact in self.list_facts():
            if fact.id == fact_id:
                return fact
        return None

    def upsert_fact(self, fact: Fact) -> str:
        """按 `fact.id` 覆盖已存在的行；`id` 为空则生成新 id 并追加。"""
        fact_id = fact.id or f"f-{uuid.uuid4().hex[:12]}"
        if fact_id != fact.id:
            fact = replace(fact, id=fact_id)

        rows = _read_jsonl(self._facts_path)
        for index, row in enumerate(rows):
            if row["id"] == fact_id:
                rows[index] = _fact_to_dict(fact)
                break
        else:
            rows.append(_fact_to_dict(fact))
        _write_jsonl(self._facts_path, rows)
        return fact_id

    def delete_fact(self, fact_id: str) -> None:
        rows = [row for row in _read_jsonl(self._facts_path) if row["id"] != fact_id]
        _write_jsonl(self._facts_path, rows)

    def list_pinned_facts(self) -> list[Fact]:
        return [
            fact for fact in self.list_facts(status="active") if fact.pinned
        ]

    def search_facts(self, query: str, *, limit: int = 20) -> list[Fact]:
        needle = query.strip().lower()
        if not needle:
            return []
        matches = [
            fact
            for fact in self.list_facts()
            if needle in fact.subject.lower()
            or needle in fact.predicate.lower()
            or needle in fact.object.lower()
            or (fact.context is not None and needle in fact.context.lower())
        ]
        matches.sort(key=lambda fact: fact.recorded_at, reverse=True)
        return matches[:limit]

    # ── entities ─────────────────────────────────────────────────────────

    def list_entities(self) -> list[Entity]:
        return [_entity_from_dict(row) for row in _read_jsonl(self._entities_path)]

    def get_entity(self, entity_id: str) -> Entity | None:
        for entity in self.list_entities():
            if entity.id == entity_id:
                return entity
        return None

    def find_entity_by_name(self, name: str) -> Entity | None:
        needle = name.strip().lower()
        for entity in self.list_entities():
            if entity.name.lower() == needle:
                return entity
            if needle in {alias.lower() for alias in entity.aliases}:
                return entity
        return None

    def upsert_entity(self, entity: Entity) -> str:
        """按 `entity.id` 覆盖已存在的行；`id` 为空则生成新 id 并追加。不做同名合并。"""
        entity_id = entity.id or f"e-{uuid.uuid4().hex[:12]}"
        if entity_id != entity.id:
            entity = replace(entity, id=entity_id)

        rows = _read_jsonl(self._entities_path)
        for index, row in enumerate(rows):
            if row["id"] == entity_id:
                rows[index] = _entity_to_dict(entity)
                break
        else:
            rows.append(_entity_to_dict(entity))
        _write_jsonl(self._entities_path, rows)
        return entity_id

    def delete_entity(self, entity_id: str) -> None:
        rows = [row for row in _read_jsonl(self._entities_path) if row["id"] != entity_id]
        _write_jsonl(self._entities_path, rows)

    def touch_entity_mention(self, entity_id: str, *, mentioned_at: str) -> None:
        entity = self.get_entity(entity_id)
        if entity is None:
            return
        updated = replace(
            entity, last_mentioned=mentioned_at, mention_count=entity.mention_count + 1
        )
        self.upsert_entity(updated)
