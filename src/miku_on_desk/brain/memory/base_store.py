"""`base` 层：原始对话单元，按日期分目录落盘为 Markdown + YAML frontmatter。

设计取舍（对应计划文档「session_id 与按日期组织的 base 层如何共存」一节）：
每个 unit 文件的 frontmatter 携带 `session_id`，`index.json` 额外维护一个 session 注册表
（`sessions`）和每个 session 的 unit 位置清单（`session_units`），外加一个全局
`unit_locations`（unit id → 相对路径）用于 O(1) 按 id 定位文件。这套 API 目前没有任何生产
调用点复用旧的高频访问路径（已用 grep 核实），因此不为假设中的高频访问做额外索引优化。

`find_semantically_similar` 用字符 bigram + ASCII 词 token 的 Jaccard 重叠做相似度启发式，
不引入向量嵌入模型——中文没有天然分词边界，bigram 是不新增依赖前提下最简单可用的近似。
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import frontmatter

from miku_on_desk.brain.memory.models import MemoryUnit, MemoryUnitRole, SessionMeta

_VALID_ROLES: frozenset[str] = frozenset({"user", "assistant", "system"})
_LINKS_FILENAME = "_links.jsonl"


class _SessionEntry(TypedDict):
    title: str
    created_at: str
    updated_at: str


class _IndexData(TypedDict):
    sessions: dict[str, _SessionEntry]
    session_units: dict[str, list[str]]
    unit_locations: dict[str, str]
    last_consolidated: str | None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _default_index() -> _IndexData:
    return {"sessions": {}, "session_units": {}, "unit_locations": {}, "last_consolidated": None}


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp-{uuid.uuid4().hex}")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _role_from_str(value: str) -> MemoryUnitRole:
    if value not in _VALID_ROLES:
        raise ValueError(f"未知的记忆单元角色：{value!r}")
    return cast(MemoryUnitRole, value)


def _token_set(text: str) -> set[str]:
    ascii_tokens = set(re.findall(r"[A-Za-z0-9]+", text.lower()))
    bigrams = {text[i : i + 2] for i in range(len(text) - 1)}
    return ascii_tokens | bigrams


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


class BaseStore:
    """`base` 层存储：一个 unit 一个 Markdown 文件 + 一份 JSON 索引。"""

    def __init__(self, root: Path, *, index_path: Path) -> None:
        self._root = root
        self._index_path = index_path
        self._root.mkdir(parents=True, exist_ok=True)

    # ── index.json 读写 ──────────────────────────────────────────────────

    def _load_index(self) -> _IndexData:
        if not self._index_path.exists():
            return _default_index()
        raw = cast(dict[str, Any], json.loads(self._index_path.read_text(encoding="utf-8")))
        index = _default_index()
        index.update(cast(_IndexData, raw))
        return index

    def _save_index(self, index: _IndexData) -> None:
        _atomic_write_text(self._index_path, json.dumps(index, ensure_ascii=False, indent=2))

    # ── unit 文件读写 ────────────────────────────────────────────────────

    def _unit_path(self, relative_path: str) -> Path:
        return self._root / relative_path

    def _read_unit_file(self, path: Path) -> MemoryUnit:
        post = frontmatter.loads(path.read_text(encoding="utf-8"))
        metadata = post.metadata
        return MemoryUnit(
            id=str(metadata["id"]),
            session_id=str(metadata["session_id"]),
            role=_role_from_str(str(metadata["role"])),
            content=post.content,
            created_at=str(metadata["created_at"]),
            model=cast(str | None, metadata.get("model")),
            provider=cast(str | None, metadata.get("provider")),
        )

    def append(self, unit: MemoryUnit) -> str:
        """写入一条新的 base 层记忆单元。若 `unit.id` 为空串则内部生成一个，返回最终 id。"""
        unit_id = unit.id or uuid.uuid4().hex
        if unit_id != unit.id:
            unit = replace(unit, id=unit_id)

        date_str = unit.created_at[:10]
        relative_path = f"{date_str}/u_{unit_id}.md"
        post = frontmatter.Post(
            unit.content,
            id=unit.id,
            session_id=unit.session_id,
            role=unit.role,
            created_at=unit.created_at,
            model=unit.model,
            provider=unit.provider,
        )
        _atomic_write_text(self._unit_path(relative_path), frontmatter.dumps(post))

        index = self._load_index()
        index["unit_locations"][unit_id] = relative_path
        index["session_units"].setdefault(unit.session_id, []).append(relative_path)
        session_entry = index["sessions"].get(unit.session_id)
        if session_entry is not None:
            session_entry["updated_at"] = unit.created_at
        self._save_index(index)
        return unit_id

    def load(self, unit_id: str) -> MemoryUnit | None:
        relative_path = self._load_index()["unit_locations"].get(unit_id)
        if relative_path is None:
            return None
        path = self._unit_path(relative_path)
        if not path.exists():
            return None
        return self._read_unit_file(path)

    def list_units(
        self, *, session_id: str | None = None, limit: int | None = None
    ) -> list[MemoryUnit]:
        index = self._load_index()
        if session_id is not None:
            relative_paths = index["session_units"].get(session_id, [])
        else:
            relative_paths = [
                path for paths in index["session_units"].values() for path in paths
            ]
            relative_paths.sort()
        units = [
            self._read_unit_file(self._unit_path(path))
            for path in relative_paths
            if self._unit_path(path).exists()
        ]
        units.sort(key=lambda unit: unit.created_at)
        if limit is not None:
            units = units[-limit:]
        return units

    def search(
        self, query: str, *, session_id: str | None = None, limit: int = 20
    ) -> list[MemoryUnit]:
        needle = query.strip().lower()
        if not needle:
            return []
        candidates = self.list_units(session_id=session_id)
        matches = [unit for unit in candidates if needle in unit.content.lower()]
        matches.sort(key=lambda unit: unit.created_at, reverse=True)
        return matches[:limit]

    # ── session 注册表 ───────────────────────────────────────────────────

    def start_session(self, session_id: str, title: str) -> None:
        index = self._load_index()
        existing = index["sessions"].get(session_id)
        if existing is not None:
            existing["title"] = title
        else:
            now = _now_iso()
            index["sessions"][session_id] = {"title": title, "created_at": now, "updated_at": now}
            index["session_units"].setdefault(session_id, [])
        self._save_index(index)

    def list_sessions(self, limit: int = 50) -> list[SessionMeta]:
        index = self._load_index()
        metas = [
            SessionMeta(
                session_id=session_id,
                title=entry["title"],
                created_at=entry["created_at"],
                updated_at=entry["updated_at"],
            )
            for session_id, entry in index["sessions"].items()
        ]
        metas.sort(key=lambda meta: meta.updated_at, reverse=True)
        return metas[:limit]

    def update_session_title(self, session_id: str, title: str) -> None:
        index = self._load_index()
        entry = index["sessions"].get(session_id)
        if entry is None:
            return
        entry["title"] = title
        self._save_index(index)

    def delete_session(self, session_id: str) -> None:
        """只从注册表摘除该 session；已落盘的 unit 文件保留（追加即真理，不做物理删除）。"""
        index = self._load_index()
        index["sessions"].pop(session_id, None)
        index["session_units"].pop(session_id, None)
        self._save_index(index)

    # ── 链接（时序 / 语义相似度） ────────────────────────────────────────

    def link_temporal(self, prev_id: str | None, unit_id: str) -> None:
        if prev_id is None:
            return
        link_line = json.dumps(
            {"type": "temporal", "from": prev_id, "to": unit_id}, ensure_ascii=False
        )
        links_path = self._root / _LINKS_FILENAME
        links_path.parent.mkdir(parents=True, exist_ok=True)
        with links_path.open("a", encoding="utf-8") as handle:
            handle.write(link_line + "\n")

    def find_semantically_similar(
        self, unit: MemoryUnit, threshold: float = 0.80
    ) -> list[tuple[str, float]]:
        query_tokens = _token_set(unit.content)
        scored: list[tuple[str, float]] = []
        for candidate in self.list_units():
            if candidate.id == unit.id:
                continue
            score = _jaccard(query_tokens, _token_set(candidate.content))
            if score >= threshold:
                scored.append((candidate.id, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    # ── 整理（consolidation）时间戳 ──────────────────────────────────────

    def get_last_consolidated(self) -> str | None:
        return self._load_index()["last_consolidated"]

    def set_last_consolidated(self, timestamp: str) -> None:
        index = self._load_index()
        index["last_consolidated"] = timestamp
        self._save_index(index)
