"""`MemorySystem` 门面：装配 base/semantic/episodic/emotional 四层存储，暴露统一入口。

`remember`/`recall` 保持 `brain/tools/builtin/memory_tools.py` 的线上工具契约：`remember`
直接写一条 `subject="user"` 的 Fact（`extracted_by="tool:remember"`），复用
`conflict.resolve_conflicts` 的同 `(subject, predicate)` 冲突解决逻辑做「同 key 更新而非
重复」。`recall` 先跑三路 `retrieval.retrieve_hints`，命中数不够 `limit` 时再从 `base` 层
原始单元补齐：`remember` 写入是同步的，本应立即可见，但普通对话内容要等异步
`extraction.run_extractions` 跑完才会进入语义/情景/情感三层，补齐 base 层能让本轮刚说过的
话在同一轮 `recall` 里就能查到。

`run_consolidation` 是夜间触发的深度整理，落地为三件确定性、可测试的工作（不做衰减计算/
向量实体链接）：
1. 对所有 `active` 事实按 `(subject, predicate)` 重新跑一遍冲突消解——覆盖“事实分批异步写入、
   跨批次的冲突未必都在写入时刻被比较过”的情况。
2. 按不区分大小写的实体名做别名合并（`find_entity_by_name` 同款粒度，不做向量/语义链接）。
3. 把 `superseded` 事实迁移到 `archived`——`archived` 状态本身就是终态，迁移只是状态位
   翻转。
完成后把时间戳写入 `base` 层 `index.json` 的 `last_consolidated`（调用方——未来 `main.py`
启动时的机会性检查——用它做 24 小时节流，本模块不关心节流策略）。
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from miku_on_desk.brain.memory import retrieval
from miku_on_desk.brain.memory.base_store import BaseStore
from miku_on_desk.brain.memory.conflict import resolve_conflicts
from miku_on_desk.brain.memory.emotional_store import EmotionalStore
from miku_on_desk.brain.memory.episodic_store import EpisodicStore
from miku_on_desk.brain.memory.models import Fact, MemoryUnit, RetrievedMemoryHint
from miku_on_desk.brain.memory.semantic_store import SemanticStore
from miku_on_desk.config.settings import EnvBootstrap, MemoryTuningConfig

logger = logging.getLogger(__name__)

_REMEMBER_SUBJECT = "user"
_REMEMBER_EXTRACTED_BY = "tool:remember"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class MemorySystem:
    """四层记忆存储的统一门面。"""

    def __init__(self, root: Path, *, tuning: MemoryTuningConfig | None = None) -> None:
        self.root = root
        self._tuning = tuning or MemoryTuningConfig()
        self.base = BaseStore(
            root / "base",
            index_path=root / "index.json",
            default_similarity_threshold=self._tuning.base_similarity_threshold,
        )
        self.semantic = SemanticStore(root / "semantic")
        self.episodic = EpisodicStore(root / "episodic")
        self.emotional = EmotionalStore(
            root / "emotional",
            default_confidence_threshold=self._tuning.emotional_confidence_threshold,
        )

    @property
    def tuning(self) -> MemoryTuningConfig:
        return self._tuning

    def add_memory_unit(self, unit: MemoryUnit) -> str:
        similar = self.base.find_semantically_similar(unit, session_id=unit.session_id)
        if similar:
            logger.debug("疑似重复记忆单元 %s，同会话内命中 %d 条相似记录", unit.id, len(similar))
        return self.base.append(unit)

    def remember(self, key: str, value: str) -> None:
        now = _now_iso()
        fact = Fact(
            id="",
            subject=_REMEMBER_SUBJECT,
            subject_type="person",
            predicate=key,
            object=value,
            object_type="concept",
            confidence=1.0,
            source=[],
            valid_from=now,
            recorded_at=now,
            extracted_by=_REMEMBER_EXTRACTED_BY,
            status="active",
        )
        new_id = self.semantic.upsert_fact(fact)
        stored = replace(fact, id=new_id)
        siblings = [
            existing
            for existing in self.semantic.list_facts(subject=_REMEMBER_SUBJECT, status="active")
            if existing.id != stored.id and existing.predicate == stored.predicate
        ]
        for winner in resolve_conflicts([stored, *siblings], semantic=self.semantic):
            self.semantic.upsert_fact(winner)

    def recall(self, query: str, limit: int = 20) -> list[RetrievedMemoryHint]:
        hints = retrieval.retrieve_hints(
            semantic=self.semantic,
            episodic=self.episodic,
            emotional=self.emotional,
            query=query,
            limit=limit,
            min_confidence=self._tuning.retrieval_min_confidence,
        )
        remaining = limit - len(hints)
        if remaining > 0:
            hints.extend(
                RetrievedMemoryHint(label="原始", text=f"{unit.role}：{unit.content}")
                for unit in self.base.search(query, limit=remaining)
            )
        return hints

    def retrieve_hints(self, query: str, limit: int = 5) -> list[RetrievedMemoryHint]:
        return retrieval.retrieve_hints(
            semantic=self.semantic,
            episodic=self.episodic,
            emotional=self.emotional,
            query=query,
            limit=limit,
            min_confidence=self._tuning.retrieval_min_confidence,
        )

    def retrieve(self, query: str, token_budget: int = 2000) -> str:
        return retrieval.retrieve(
            semantic=self.semantic,
            episodic=self.episodic,
            emotional=self.emotional,
            query=query,
            token_budget=token_budget,
            min_confidence=self._tuning.retrieval_min_confidence,
        )

    def run_consolidation(self, *, now: str | None = None) -> None:
        self._resolve_all_conflicts()
        self._merge_duplicate_entities()
        self._archive_superseded_facts()
        self.base.set_last_consolidated(now or _now_iso())

    def _resolve_all_conflicts(self) -> None:
        groups: dict[tuple[str, str], list[Fact]] = {}
        for fact in self.semantic.list_facts(status="active"):
            groups.setdefault((fact.subject, fact.predicate), []).append(fact)
        for group in groups.values():
            if len(group) < 2:
                continue
            for winner in resolve_conflicts(group, semantic=self.semantic):
                self.semantic.upsert_fact(winner)

    def _merge_duplicate_entities(self) -> None:
        groups: dict[str, list[str]] = {}
        for entity in self.semantic.list_entities():
            groups.setdefault(entity.name.strip().lower(), []).append(entity.id)
        for entity_ids in groups.values():
            if len(entity_ids) < 2:
                continue
            entities = [self.semantic.get_entity(entity_id) for entity_id in entity_ids]
            duplicates = [entity for entity in entities if entity is not None]
            duplicates.sort(key=lambda entity: entity.first_seen or entity.last_mentioned)
            canonical = duplicates[0]
            for other in duplicates[1:]:
                aliases = list(canonical.aliases)
                for alias in [other.name, *other.aliases]:
                    if alias != canonical.name and alias not in aliases:
                        aliases.append(alias)
                canonical = replace(
                    canonical,
                    aliases=aliases,
                    last_mentioned=max(canonical.last_mentioned, other.last_mentioned),
                    mention_count=canonical.mention_count + other.mention_count,
                )
                self.semantic.delete_entity(other.id)
            self.semantic.upsert_entity(canonical)

    def _archive_superseded_facts(self) -> None:
        for fact in self.semantic.list_facts(status="superseded"):
            self.semantic.upsert_fact(replace(fact, status="archived"))


def default_memory_system(
    memory_dir: Path | None,
    bootstrap: EnvBootstrap | None = None,
    *,
    tuning: MemoryTuningConfig | None = None,
) -> MemorySystem:
    bootstrap = bootstrap or EnvBootstrap()
    resolved = memory_dir if memory_dir is not None else bootstrap.resolve_data_dir() / "memory"
    resolved.mkdir(parents=True, exist_ok=True)
    return MemorySystem(resolved, tuning=tuning)
