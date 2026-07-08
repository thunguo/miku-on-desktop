"""四层文件系统记忆架构的核心数据形状：`base`/`semantic`/`episodic`/`emotional` 共用。

`Fact.pinned` 是新增字段：旧 SQLite 实现里 `core` 记忆层承担"常驻 frozen system prompt"
这一体验，用这个布尔位在新架构里保留同等能力（见 `semantic_store.list_pinned_facts`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MemoryUnitRole = Literal["user", "assistant", "system"]
EntityType = Literal["person", "location", "organization", "concept", "event", "technology"]
FactStatus = Literal["active", "superseded", "conflict", "archived"]


@dataclass(frozen=True)
class MemoryUnit:
    """`base` 层的一条原始对话记录，对应磁盘上的一个 `base/YYYY-MM-DD/u_<id>.md` 文件。"""

    id: str
    session_id: str
    role: MemoryUnitRole
    content: str
    created_at: str
    model: str | None = None
    provider: str | None = None


@dataclass(frozen=True)
class Fact:
    """`semantic` 层的一条事实三元组，对应 `semantic/facts.jsonl` 的一行。"""

    id: str
    subject: str
    subject_type: EntityType
    predicate: str
    object: str
    object_type: EntityType
    confidence: float
    source: list[str]
    valid_from: str
    recorded_at: str
    extracted_by: str
    status: FactStatus
    context: str | None = None
    pinned: bool = False


@dataclass(frozen=True)
class Entity:
    """`semantic` 层的一个实体定义，对应 `semantic/entities.jsonl` 的一行。"""

    id: str
    name: str
    type: EntityType
    aliases: list[str] = field(default_factory=list)
    first_seen: str = ""
    last_mentioned: str = ""
    mention_count: int = 0


@dataclass(frozen=True)
class Episode:
    """`episodic` 层的一个事件块，对应 `episodic/YYYY/YYYY-MM.md` 里的一个 `### [E:NNN]` 块。

    「事件链」子序列、「关联事件」跨事件引用是叙事性自由文本，不是任何编排逻辑会结构化生成/
    消费的字段，这里各自建模成 `list[str]`（每行一条），照原样落盘/读回，不做进一步解析。
    """

    id: str
    month: str
    title: str
    occurred_at: str
    source_units: list[str] = field(default_factory=list)
    summary: str = ""
    emotion_tag: str | None = None
    participants: list[str] = field(default_factory=list)
    event_chain: list[str] = field(default_factory=list)
    related_events: list[str] = field(default_factory=list)
    session_id: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class RetrievedMemoryHint:
    label: str
    text: str


@dataclass(frozen=True)
class SessionMeta:
    """`index.json` 里 session 注册表的一条记录（unit 清单由 `BaseStore` 内部单独管理）。"""

    session_id: str
    title: str
    created_at: str
    updated_at: str
