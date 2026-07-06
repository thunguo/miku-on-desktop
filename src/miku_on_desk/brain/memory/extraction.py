"""提取管线：把 `base` 层新增的对话单元，异步提炼进 semantic/episodic/emotional 三层。

设计文档 §7.1 伪代码用 `asyncio.gather` 并发跑三个子提取器；这里做了一处调度节奏上的针对性
简化：语义/情感两路提取器保持文档说的"即时触发"语义，每次 `run_extractions` 调用都会跑；
情景提取按计划文档确认的"每 6 轮对话或 10 分钟"节奏批量触发。§6.2 的
`.tmp/pending_extractions.jsonl` 队列在本模块里只承担"情景批次累计到第几轮/最早一轮是什么
时候"这一调度状态的跨进程重启持久化——语义/情感两路每次调用都立即处理完毕，不需要跨调用
排队，因此不占用这个队列。

三个子提取器各自独立失败：某一路 LLM 调用/JSON 解析失败，只跳过那一路的产出，不影响另外两路，
也不向上抛异常中断整个 `run_extractions` 调用——只检查 `result.success`，不包 try/except；
是否需要"一次提取失败不能打断整轮对话"这层兜底，是调用方（`main.py` 的背景任务包装函数）的
职责，本模块不重复包一层。
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from miku_on_desk.brain.memory.base_store import BaseStore
from miku_on_desk.brain.memory.conflict import resolve_conflicts
from miku_on_desk.brain.memory.emotional_store import EmotionalStore
from miku_on_desk.brain.memory.episodic_store import EpisodicStore
from miku_on_desk.brain.memory.models import EntityType, Fact, MemoryUnit
from miku_on_desk.brain.memory.semantic_store import SemanticStore
from miku_on_desk.brain.model_router import ModelRouter
from miku_on_desk.brain.providers.base import Message, Provider
from miku_on_desk.config.settings import ModelTier, ProviderName

_EXTRACTION_TIER = ModelTier.FAST
_PENDING_FILENAME = "pending_extractions.jsonl"
_EPISODIC_BATCH_TURNS = 6
_EPISODIC_BATCH_SECONDS = 10 * 60

_VALID_ENTITY_TYPES: frozenset[str] = frozenset(
    {"person", "location", "organization", "concept", "event", "technology"}
)

_SEMANTIC_SYSTEM_PROMPT = (
    "你负责从一段对话里提炼可验证的事实三元组，只提取用户明确陈述或强烈暗示的稳定信息，"
    "忽略任务性/临时性的内容。"
    '严格输出 JSON：{"facts": [{"subject": "用户", "subject_type": "person", '
    '"predicate": "住在", "object": "上海", "object_type": "location", '
    '"confidence": 0.9}]}，没有新信息就输出空列表，不要输出任何其他文字。'
)

_EPISODIC_SYSTEM_PROMPT = (
    "你负责把一段时间内的对话整理成一条情景记忆事件：给出简短标题、原始摘要，以及可选的"
    "情感标记、参与实体、事件链条目。"
    '严格输出 JSON：{"title": "...", "summary": "...", "emotion_tag": "温暖", '
    '"participants": ["Lisa"], "event_chain": ["提到搬家"]}，emotion_tag 没有就输出 '
    "null，其余没有就输出空列表，不要输出任何其他文字。"
)

_EMOTIONAL_SYSTEM_PROMPT = (
    "你负责维护用户的情感/偏好档案（JSON），根据新对话内容判断是否需要更新其中的叶子字段。"
    "只输出需要新增/修改的叶子路径，不要重复未变化的内容。"
    '严格输出 JSON：{"updates": {"location_preferences.familiar_cities": [...]}}'
    '（路径用 "a.b.c" 形式表示嵌套），没有需要更新的就输出 {"updates": {}}，不要输出任何'
    "其他文字。"
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _pending_path(root: Path) -> Path:
    return root / ".tmp" / _PENDING_FILENAME


def _read_pending(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [cast(dict[str, Any], json.loads(line)) for line in lines if line.strip()]


def _write_pending(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp-{uuid.uuid4().hex}")
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    tmp_path.write_text(content + "\n" if rows else "", encoding="utf-8")
    os.replace(tmp_path, path)


def _should_flush_episodic(pending: list[dict[str, Any]], *, now: str) -> bool:
    if len(pending) >= _EPISODIC_BATCH_TURNS:
        return True
    if not pending:
        return False
    oldest = min(cast(str, row["created_at"]) for row in pending)
    elapsed = (datetime.fromisoformat(now) - datetime.fromisoformat(oldest)).total_seconds()
    return elapsed >= _EPISODIC_BATCH_SECONDS


def _format_units(units: Sequence[MemoryUnit]) -> str:
    return "\n".join(f"{unit.role}：{unit.content}" for unit in units)


def _entity_type_or_default(value: object) -> EntityType:
    if isinstance(value, str) and value in _VALID_ENTITY_TYPES:
        return cast(EntityType, value)
    return "concept"


def _parse_semantic_facts(text: str) -> list[dict[str, Any]]:
    try:
        data: Any = json.loads(text)
        facts = data.get("facts", [])
        return [
            row
            for row in facts
            if isinstance(row, dict)
            and row.get("subject")
            and row.get("predicate")
            and row.get("object")
        ]
    except (json.JSONDecodeError, AttributeError, TypeError, KeyError):
        return []


def _parse_episodic_event(text: str) -> dict[str, Any] | None:
    try:
        data: Any = json.loads(text)
        if not data.get("title") or not data.get("summary"):
            return None
        return cast(dict[str, Any], data)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None


def _parse_emotional_updates(text: str) -> dict[str, Any]:
    try:
        data: Any = json.loads(text)
        updates = data.get("updates", {})
        return cast(dict[str, Any], updates) if isinstance(updates, dict) else {}
    except (json.JSONDecodeError, AttributeError, TypeError):
        return {}


def _set_by_path(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = root
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            existing = {}
            cursor[part] = existing
        cursor = existing
    cursor[parts[-1]] = value


async def _extract_semantic(
    units: Sequence[MemoryUnit],
    *,
    semantic: SemanticStore,
    router: ModelRouter,
    providers: dict[ProviderName, Provider],
) -> None:
    resolved = router.resolve(_EXTRACTION_TIER)
    provider = providers[resolved.provider]
    message = Message(role="user", content=_format_units(units))
    result = await provider.stream(
        model=resolved.model_id, system=_SEMANTIC_SYSTEM_PROMPT, messages=[message], tools=[]
    )
    if not result.success or not result.content:
        return

    now = _now_iso()
    source = [unit.id for unit in units]
    for row in _parse_semantic_facts(result.content):
        fact = Fact(
            id="",
            subject=row["subject"],
            subject_type=_entity_type_or_default(row.get("subject_type")),
            predicate=row["predicate"],
            object=row["object"],
            object_type=_entity_type_or_default(row.get("object_type")),
            confidence=float(row.get("confidence", 0.7)),
            source=source,
            valid_from=now,
            recorded_at=now,
            extracted_by=f"llm:{resolved.model_id}",
            status="active",
        )
        new_id = semantic.upsert_fact(fact)
        stored = replace(fact, id=new_id)
        siblings = [
            existing
            for existing in semantic.list_facts(subject=stored.subject, status="active")
            if existing.id != stored.id and existing.predicate == stored.predicate
        ]
        for winner in resolve_conflicts([stored, *siblings], semantic=semantic):
            semantic.upsert_fact(winner)


async def _extract_episodic(
    pending: list[dict[str, Any]],
    *,
    base: BaseStore,
    episodic: EpisodicStore,
    router: ModelRouter,
    providers: dict[ProviderName, Provider],
) -> None:
    unit_ids = [unit_id for row in pending for unit_id in cast(list[str], row["source_units"])]
    units = [unit for unit_id in unit_ids for unit in [base.load(unit_id)] if unit is not None]
    if not units:
        return

    resolved = router.resolve(_EXTRACTION_TIER)
    provider = providers[resolved.provider]
    message = Message(role="user", content=_format_units(units))
    result = await provider.stream(
        model=resolved.model_id, system=_EPISODIC_SYSTEM_PROMPT, messages=[message], tools=[]
    )
    if not result.success or not result.content:
        return

    parsed = _parse_episodic_event(result.content)
    if parsed is None:
        return

    session_id = cast(str, pending[-1]["session_id"])
    episodic.append_event(
        title=parsed["title"],
        summary=parsed["summary"],
        occurred_at=units[-1].created_at,
        source_units=unit_ids,
        emotion_tag=parsed.get("emotion_tag"),
        participants=list(parsed.get("participants", [])),
        event_chain=list(parsed.get("event_chain", [])),
        session_id=session_id,
        model=resolved.model_id,
    )


async def _extract_emotional(
    units: Sequence[MemoryUnit],
    *,
    emotional: EmotionalStore,
    router: ModelRouter,
    providers: dict[ProviderName, Provider],
) -> None:
    resolved = router.resolve(_EXTRACTION_TIER)
    provider = providers[resolved.provider]
    current = emotional.load_preferences()
    profile_json = json.dumps(current, ensure_ascii=False)
    message = Message(
        role="user",
        content=f"当前偏好档案：{profile_json}\n\n{_format_units(units)}",
    )
    result = await provider.stream(
        model=resolved.model_id, system=_EMOTIONAL_SYSTEM_PROMPT, messages=[message], tools=[]
    )
    if not result.success or not result.content:
        return

    updates = _parse_emotional_updates(result.content)
    if not updates:
        return

    merged = dict(current)
    for path, value in updates.items():
        _set_by_path(merged, path, value)
    merged["last_updated"] = _now_iso()
    emotional.save_preferences(merged)


async def run_extractions(
    *,
    base: BaseStore,
    semantic: SemanticStore,
    episodic: EpisodicStore,
    emotional: EmotionalStore,
    root: Path,
    session_id: str,
    units: Sequence[MemoryUnit],
    router: ModelRouter,
    providers: dict[ProviderName, Provider],
    now: str | None = None,
) -> None:
    """给这一轮新增的 base 单元跑一次提取：语义/情感即时触发，情景按批次触发。

    `now` 默认取当前时间；测试传入固定值以让"6 轮或 10 分钟"的批次判定可复现。
    """
    if not units:
        return
    resolved_now = now or _now_iso()

    path = _pending_path(root)
    pending = _read_pending(path)
    pending.append(
        {
            "task_id": uuid.uuid4().hex,
            "type": "episodic_extraction",
            "session_id": session_id,
            "source_units": [unit.id for unit in units],
            "priority": "normal",
            "created_at": resolved_now,
            "status": "pending",
        }
    )
    should_flush = _should_flush_episodic(pending, now=resolved_now)

    tasks = [
        _extract_semantic(units, semantic=semantic, router=router, providers=providers),
        _extract_emotional(units, emotional=emotional, router=router, providers=providers),
    ]
    if should_flush:
        tasks.append(
            _extract_episodic(
                pending, base=base, episodic=episodic, router=router, providers=providers
            )
        )

    await asyncio.gather(*tasks)

    _write_pending(path, [] if should_flush else pending)
