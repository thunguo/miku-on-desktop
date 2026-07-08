"""跨格式混合检索：语义/情景/情感三路查询 + token 预算组装。

两个入口，服务不同调用方：
- `retrieve_hints`：每轮对话都会调用的轻量版本（`reminder.py` 用），只做三路关键词搜索 +
  排序截断，不做 token 预算组装——每轮提醒只需要几行提示，不是一整块可读上下文。
- `retrieve`：完整读取管线，按语义 40% / 情景 30% / 情感 15%（剩余 15% 是系统提示 + 格式
  开销的预留份额，不由本函数消费）分配 `token_budget`，拼成一段可读文本块。

两处针对性简化：
1. 不做查询分析与意图分类（实体识别/意图分类/查询向量化）——直接把原始查询字符串同时喂给
   三路子检索，省掉一个需要额外 LLM 调用才能做的查询理解步骤；三路各自的子串匹配已经
   对齐 `semantic_store.search_facts`/`episodic_store.search` 现有实现。
2. 不在读取路径里跑 `conflict.resolve_conflicts`——`extraction.py` 已经在写入时把同一
   `(subject, predicate)` 分组的冲突事实解决掉（败者标记 `superseded`），读取路径只需要按
   `status == "active"` 过滤即可，不需要在每次检索时重新计算一遍冲突分组。`SemanticStore.
   search_facts`/`list_facts` 本身不按 `status` 过滤（它们是哑存储层），所以这里显式补上
   这一步。
"""

from __future__ import annotations

import json
from typing import Any

from miku_on_desk.brain.memory.emotional_store import EmotionalStore
from miku_on_desk.brain.memory.episodic_store import EpisodicStore
from miku_on_desk.brain.memory.models import RetrievedMemoryHint
from miku_on_desk.brain.memory.semantic_store import SemanticStore

_CANDIDATE_FETCH_LIMIT = 30
_MIN_CONFIDENCE = 0.7

_DEFAULT_TOKEN_BUDGET = 2000
_SEMANTIC_BUDGET_RATIO = 0.40
_EPISODIC_BUDGET_RATIO = 0.30
_EMOTIONAL_BUDGET_RATIO = 0.15
_CHARS_PER_TOKEN_ESTIMATE = 2
"""与 `compaction.py` 的同名估算比例保持一致，避免同一份文本在两处估出不同 token 数。"""

_PREFERENCES_METADATA_KEYS = frozenset({"version", "last_updated"})


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN_ESTIMATE


def _truncate_to_budget(lines: list[str], *, budget_tokens: int) -> list[str]:
    kept: list[str] = []
    used = 0
    for line in lines:
        cost = _estimate_tokens(line)
        if used + cost > budget_tokens:
            break
        kept.append(line)
        used += cost
    return kept


def _flatten_preferences(data: Any, *, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(data, dict):
        leaves: list[tuple[str, Any]] = []
        for key, value in data.items():
            if not prefix and key in _PREFERENCES_METADATA_KEYS:
                continue
            path = f"{prefix}.{key}" if prefix else key
            leaves.extend(_flatten_preferences(value, prefix=path))
        return leaves
    return [(prefix, data)]


def _render_leaf(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _search_emotional(
    emotional: EmotionalStore, query: str, *, limit: int
) -> list[tuple[str, str]]:
    needle = query.strip().lower()
    if not needle:
        return []
    leaves = _flatten_preferences(emotional.load_preferences())
    matches: list[tuple[str, str]] = []
    for path, value in leaves:
        rendered = _render_leaf(value)
        if needle in path.lower() or needle in rendered.lower():
            matches.append((path, rendered))
    return matches[:limit]


def retrieve_hints(
    *,
    semantic: SemanticStore,
    episodic: EpisodicStore,
    emotional: EmotionalStore,
    query: str,
    limit: int = 5,
) -> list[RetrievedMemoryHint]:
    """轻量三路检索：语义事实 + 情景标题摘要 + 情感偏好，按此优先级截断到 `limit` 条。"""
    active_facts = [
        fact
        for fact in semantic.search_facts(query, limit=_CANDIDATE_FETCH_LIMIT)
        if fact.status == "active"
    ]
    hints = [
        RetrievedMemoryHint(label="语义", text=f"{fact.subject}{fact.predicate}{fact.object}")
        for fact in active_facts[:limit]
    ]
    hints.extend(
        RetrievedMemoryHint(label="情景", text=f"{episode.title}：{episode.summary}")
        for episode in episodic.search(query, limit=limit)
    )
    hints.extend(
        RetrievedMemoryHint(label="情感", text=f"{path}：{value}")
        for path, value in _search_emotional(emotional, query, limit=limit)
    )
    return hints[:limit]


def retrieve(
    *,
    semantic: SemanticStore,
    episodic: EpisodicStore,
    emotional: EmotionalStore,
    query: str,
    token_budget: int = _DEFAULT_TOKEN_BUDGET,
    min_confidence: float = _MIN_CONFIDENCE,
) -> str:
    """完整读取管线：三路检索，按 40/30/15% 的 token 预算拼成一段可读文本。"""
    active_facts = [
        fact
        for fact in semantic.search_facts(query, limit=_CANDIDATE_FETCH_LIMIT)
        if fact.status == "active" and fact.confidence > min_confidence
    ]
    active_facts.sort(key=lambda fact: fact.confidence, reverse=True)
    episodes = episodic.search(query, limit=_CANDIDATE_FETCH_LIMIT)
    emotional_matches = _search_emotional(emotional, query, limit=_CANDIDATE_FETCH_LIMIT)

    sections: list[str] = []

    semantic_lines = _truncate_to_budget(
        [
            f"- {fact.subject}{fact.predicate}{fact.object}（置信度 {fact.confidence:.2f}）"
            for fact in active_facts
        ],
        budget_tokens=int(token_budget * _SEMANTIC_BUDGET_RATIO),
    )
    if semantic_lines:
        sections.append("已知事实：\n" + "\n".join(semantic_lines))

    episodic_lines = _truncate_to_budget(
        [f"- [{episode.occurred_at}] {episode.title}：{episode.summary}" for episode in episodes],
        budget_tokens=int(token_budget * _EPISODIC_BUDGET_RATIO),
    )
    if episodic_lines:
        sections.append("相关事件：\n" + "\n".join(episodic_lines))

    emotional_lines = _truncate_to_budget(
        [f"- {path}：{value}" for path, value in emotional_matches],
        budget_tokens=int(token_budget * _EMOTIONAL_BUDGET_RATIO),
    )
    if emotional_lines:
        sections.append("偏好参考：\n" + "\n".join(emotional_lines))

    return "\n\n".join(sections)
