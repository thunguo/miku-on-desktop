"""跨事实的冲突检测与消解。

两处针对本项目实际 schema 的必要简化：
1. Fact 只有 `valid_from`（时间点），没有"有效期区间"的 `valid_to`（区间终点）——本次不新增
   `valid_to` 字段，把"时间重叠"简化为"既有事实是否仍处于 active 状态、且早于新事实"这一
   可判定的点时间近似（见 `_temporal_overlap`）。
2. Fact schema 没有 `superseded_by` 反向引用字段（只有 `status` 状态机），标记被取代事实
   直接用 `SemanticStore.upsert_fact` 把败者的 `status` 原地改写成 `superseded`，不新增字段。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Literal

from miku_on_desk.brain.memory.models import Fact
from miku_on_desk.brain.memory.semantic_store import SemanticStore

ConflictType = Literal["value_conflict", "temporal_conflict"]

_TEMPORAL_PREDICATE_KEYWORDS = ("住", "居住", "工作", "任职", "就读", "搬迁", "搬到", "位于")


@dataclass(frozen=True)
class Conflict:
    """一次冲突检测命中。"""

    type: ConflictType
    existing: Fact
    new: Fact
    resolution: str


def _is_temporal_predicate(predicate: str) -> bool:
    return any(keyword in predicate for keyword in _TEMPORAL_PREDICATE_KEYWORDS)


def _temporal_overlap(existing: Fact, new: Fact) -> bool:
    return existing.status == "active" and existing.valid_from <= new.valid_from


def detect_conflicts(new_fact: Fact, existing_facts: list[Fact]) -> list[Conflict]:
    """同主谓不同宾语的取值冲突 + 时间性谓语的时间冲突。"""
    conflicts: list[Conflict] = []
    for fact in existing_facts:
        if fact.status != "active" or fact.id == new_fact.id:
            continue
        if (
            fact.subject == new_fact.subject
            and fact.predicate == new_fact.predicate
            and fact.object != new_fact.object
        ):
            conflicts.append(
                Conflict(
                    type="value_conflict",
                    existing=fact,
                    new=new_fact,
                    resolution="higher_confidence_wins",
                )
            )
        if (
            fact.subject == new_fact.subject
            and _is_temporal_predicate(fact.predicate)
            and _temporal_overlap(fact, new_fact)
        ):
            conflicts.append(
                Conflict(
                    type="temporal_conflict",
                    existing=fact,
                    new=new_fact,
                    resolution="latest_valid_wins",
                )
            )
    return conflicts


def resolve_conflicts(facts: list[Fact], *, semantic: SemanticStore) -> list[Fact]:
    """按 (subject, predicate) 分组，高置信度优先、僵持时取更新者胜出。"""
    groups: dict[tuple[str, str], list[Fact]] = defaultdict(list)
    for fact in facts:
        groups[(fact.subject, fact.predicate)].append(fact)

    resolved: list[Fact] = []
    for group in groups.values():
        if len(group) == 1:
            resolved.append(group[0])
            continue

        best = max(group, key=lambda f: f.confidence)
        for fact in group:
            if fact.id != best.id and abs(fact.confidence - best.confidence) < 0.1:
                best = max([best, fact], key=lambda f: f.valid_from)

        for fact in group:
            if fact.id != best.id:
                semantic.upsert_fact(replace(fact, status="superseded"))

        resolved.append(best)
    return resolved
