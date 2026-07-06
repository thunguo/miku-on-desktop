"""`episodic` 层：按月 Markdown 文件存储的事件链，格式对齐设计文档 §4.1 worked example。

显式偏离设计文档的两点（在计划文档「与设计文档的显式偏离」一节之外，属本模块局部实现细节，
一并记录在这里）：
1. 不做 `## Week NN (...)` 周分组——纯展示性分组，对检索/溯源没有功能价值，反而让块解析
   复杂化，直接把所有 `### [E:NNN]` 事件块平铺在月标题下。
2. `Episode` 在设计文档 schema 之外新增了 `session_id`/`model` 两个可选字段（渲染为
   `**来源会话**`/`**生成模型**` 行，缺省不渲染），用于 `compaction.py` 把压缩摘要落盘时
   保留"这条事件来自哪个会话/由哪个模型生成"的可追溯性——跟 `Fact.pinned` 是同一类"为保留
   现有能力而做的针对性加字段"处理。

事件链（`事件链`/`关联事件`）在设计文档里是叙事性自由文本，本次没有任何编排逻辑会结构化生成
或消费这两个子字段（`memory_panel.py` 情景标签页也明确不做这两个子字段的编辑表单），因此
建模成 `list[str]`（每行一条），原样落盘/读回，不做进一步语义解析。
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import replace
from pathlib import Path

import frontmatter

from miku_on_desk.brain.memory.models import Episode

_EVENT_HEADER_RE = re.compile(r"^### \[(E:\d+)\] (.*)$")
_EVENT_ID_RE = re.compile(r"### \[E:(\d+)\]")
_INDEX_FILENAME = "_index.md"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp-{uuid.uuid4().hex}")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _month_title(month: str) -> str:
    year, _, mon = month.partition("-")
    return f"{year}年{int(mon)}月 —— 事件链"


def _split_blocks(body: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in body.splitlines():
        if _EVENT_HEADER_RE.match(line):
            if current is not None:
                blocks.append(current)
            current = [line]
        elif current is not None:
            current.append(line)
    if current is not None:
        blocks.append(current)
    return blocks


def _parse_event_block(lines: list[str], month: str) -> Episode:
    header_match = _EVENT_HEADER_RE.match(lines[0])
    assert header_match is not None
    event_id = header_match.group(1)
    title = header_match.group(2).strip()

    occurred_at = ""
    source_units: list[str] = []
    emotion_tag: str | None = None
    participants: list[str] = []
    event_chain: list[str] = []
    summary_lines: list[str] = []
    related_events: list[str] = []
    session_id: str | None = None
    model: str | None = None
    section: str | None = None

    for raw_line in lines[1:]:
        text = raw_line.strip()
        if text.startswith("**时间**:"):
            occurred_at = text.split(":", 1)[1].strip()
            section = None
        elif text.startswith("**来源单元**:"):
            raw = text.split(":", 1)[1].strip()
            source_units = [item.strip() for item in raw.split(",") if item.strip()]
            section = None
        elif text.startswith("**情感标记**:"):
            emotion_tag = text.split(":", 1)[1].strip() or None
            section = None
        elif text.startswith("**参与实体**:"):
            raw = text.split(":", 1)[1].strip()
            participants = re.findall(r"\[\[([^\]]+)\]\]", raw)
            section = None
        elif text.startswith("**来源会话**:"):
            session_id = text.split(":", 1)[1].strip() or None
            section = None
        elif text.startswith("**生成模型**:"):
            model = text.split(":", 1)[1].strip() or None
            section = None
        elif text.startswith("**事件链**"):
            section = "event_chain"
        elif text.startswith("**原始摘要**"):
            section = "summary"
        elif text.startswith("**关联事件**"):
            section = "related"
        elif text == "---" or not text:
            continue
        elif section == "event_chain":
            event_chain.append(re.sub(r"^\d+\.\s*", "", text))
        elif section == "summary":
            summary_lines.append(re.sub(r"^>\s?", "", text))
        elif section == "related":
            related_events.append(re.sub(r"^-\s*", "", text))

    return Episode(
        id=event_id,
        month=month,
        title=title,
        occurred_at=occurred_at,
        source_units=source_units,
        summary=" ".join(summary_lines).strip(),
        emotion_tag=emotion_tag,
        participants=participants,
        event_chain=event_chain,
        related_events=related_events,
        session_id=session_id,
        model=model,
    )


def _render_event_block(episode: Episode) -> str:
    lines = [f"### [{episode.id}] {episode.title}", f"**时间**: {episode.occurred_at}  "]
    lines.append(f"**来源单元**: {', '.join(episode.source_units)}  ")
    if episode.emotion_tag:
        lines.append(f"**情感标记**: {episode.emotion_tag}  ")
    if episode.participants:
        wiki_links = ", ".join(f"[[{name}]]" for name in episode.participants)
        lines.append(f"**参与实体**: {wiki_links}  ")
    if episode.session_id:
        lines.append(f"**来源会话**: {episode.session_id}  ")
    if episode.model:
        lines.append(f"**生成模型**: {episode.model}  ")
    lines.append("")
    if episode.event_chain:
        lines.append("**事件链**:")
        lines.extend(f"{index}. {item}" for index, item in enumerate(episode.event_chain, 1))
        lines.append("")
    lines.append("**原始摘要**:  ")
    lines.append(f"> {episode.summary}")
    if episode.related_events:
        lines.append("")
        lines.append("**关联事件**:")
        lines.extend(f"- {item}" for item in episode.related_events)
    lines.append("")
    lines.append("---")
    return "\n".join(lines)


class EpisodicStore:
    """`episodic` 层存储：按月 Markdown 文件 + 一份全局 `_index.md` 目录。"""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def _month_path(self, month: str) -> Path:
        year = month[:4]
        return self._root / year / f"{month}.md"

    def _all_month_paths(self) -> list[Path]:
        return sorted(self._root.glob("*/*.md"))

    def _read_month(self, path: Path) -> tuple[dict[str, object], list[list[str]]]:
        if not path.exists():
            return {}, []
        post = frontmatter.loads(path.read_text(encoding="utf-8"))
        return dict(post.metadata), _split_blocks(post.content)

    def _write_month(
        self, path: Path, month: str, blocks: list[list[str]], *, last_compacted: str
    ) -> None:
        body_parts = [f"# {_month_title(month)}", ""]
        body_parts.extend("\n".join(block) + "\n" for block in blocks)
        post = frontmatter.Post(
            "\n".join(body_parts),
            version="1.0",
            month=month,
            generated_by="miku-on-desk",
            last_compacted=last_compacted,
        )
        _atomic_write_text(path, frontmatter.dumps(post))

    def _next_event_id(self) -> str:
        max_number = 0
        for path in self._all_month_paths():
            for match in _EVENT_ID_RE.finditer(path.read_text(encoding="utf-8")):
                max_number = max(max_number, int(match.group(1)))
        return f"E:{max_number + 1:03d}"

    def _find_location(self, event_id: str) -> tuple[Path, list[list[str]], int] | None:
        for path in self._all_month_paths():
            _metadata, blocks = self._read_month(path)
            for index, block in enumerate(blocks):
                header_match = _EVENT_HEADER_RE.match(block[0])
                if header_match is not None and header_match.group(1) == event_id:
                    return path, blocks, index
        return None

    def append_event(
        self,
        *,
        title: str,
        summary: str,
        occurred_at: str,
        source_units: list[str] | None = None,
        emotion_tag: str | None = None,
        participants: list[str] | None = None,
        event_chain: list[str] | None = None,
        related_events: list[str] | None = None,
        session_id: str | None = None,
        model: str | None = None,
    ) -> str:
        event_id = self._next_event_id()
        month = occurred_at[:7]
        episode = Episode(
            id=event_id,
            month=month,
            title=title,
            occurred_at=occurred_at,
            source_units=list(source_units or []),
            summary=summary,
            emotion_tag=emotion_tag,
            participants=list(participants or []),
            event_chain=list(event_chain or []),
            related_events=list(related_events or []),
            session_id=session_id,
            model=model,
        )
        path = self._month_path(month)
        _metadata, blocks = self._read_month(path)
        blocks.append(_render_event_block(episode).splitlines())
        self._write_month(path, month, blocks, last_compacted=occurred_at)
        self._rebuild_index()
        return event_id

    def get_event(self, event_id: str) -> Episode | None:
        location = self._find_location(event_id)
        if location is None:
            return None
        path, blocks, index = location
        month = path.stem
        return _parse_event_block(blocks[index], month)

    def list_events(self, *, month: str | None = None, limit: int | None = None) -> list[Episode]:
        if month is not None:
            _metadata, blocks = self._read_month(self._month_path(month))
            episodes = [_parse_event_block(block, month) for block in blocks]
        else:
            episodes = []
            for path in self._all_month_paths():
                _metadata, blocks = self._read_month(path)
                episodes.extend(_parse_event_block(block, path.stem) for block in blocks)
        episodes.sort(key=lambda episode: episode.occurred_at)
        if limit is not None:
            episodes = episodes[-limit:]
        return episodes

    def list_months(self) -> list[str]:
        return sorted(path.stem for path in self._all_month_paths())

    def search(self, query: str, *, limit: int = 20) -> list[Episode]:
        needle = query.strip().lower()
        if not needle:
            return []
        matches = [
            episode
            for episode in self.list_events()
            if needle in episode.title.lower()
            or needle in episode.summary.lower()
            or any(needle in name.lower() for name in episode.participants)
            or any(needle in item.lower() for item in episode.event_chain)
        ]
        matches.sort(key=lambda episode: episode.occurred_at, reverse=True)
        return matches[:limit]

    def update_summary(self, event_id: str, summary: str) -> None:
        location = self._find_location(event_id)
        if location is None:
            return
        path, blocks, index = location
        month = path.stem
        episode = replace(_parse_event_block(blocks[index], month), summary=summary)
        blocks[index] = _render_event_block(episode).splitlines()
        self._write_month(path, month, blocks, last_compacted=episode.occurred_at)
        self._rebuild_index()

    def delete_event(self, event_id: str) -> None:
        location = self._find_location(event_id)
        if location is None:
            return
        path, blocks, index = location
        month = path.stem
        del blocks[index]
        self._write_month(path, month, blocks, last_compacted=month)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        episodes = self.list_events()
        lines = ["# 情景记忆索引", ""]
        for episode in episodes:
            lines.append(
                f"- [{episode.id}] {episode.title} — {episode.occurred_at} "
                f"(episodic/{episode.month[:4]}/{episode.month}.md)"
            )
        _atomic_write_text(self._root / _INDEX_FILENAME, "\n".join(lines) + "\n")
