"""冻结系统提示：把变化频率不同的静态内容按固定顺序拼接，最大化 prompt cache 命中率。

系统提示是 Anthropic prompt cache 里命中率最高、收益最大的一个断点，只要这段文本的字节序列
在会话内保持稳定，同一会话的后续请求都能免费复用它。为了尽量延长"稳定"的窗口，各分区按变化
频率从低到高排列——agents 几乎不变放最前，skills 靠文件监听器触发变化放第二，记忆索引随记忆
工具写入而变，核心记忆内容变化最频繁放最后。这样安排的意义是：任何一次变化只会作废这段
字节序列的尾部，前面大段静态内容的缓存前缀始终保持命中。

各分区的具体内容来自 skills/agents/memory 子系统，本模块不反向依赖那些具体实现——只接受
它们产出的纯字符串摘要，不需要改动这里的拼接逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass

_SECTION_ORDER: tuple[tuple[str, str], ...] = (
    ("agents_summary", "已启用的 Sub-agent"),
    ("skills_summary", "已启用的 Skills"),
    ("memory_index_summary", "记忆索引"),
    ("core_memory", "核心记忆"),
)


@dataclass(frozen=True)
class FrozenSystemSections:
    """``identity`` 是恒定不变的基础人设/指令，永远排在最前；其余分区留空即视为不存在。"""

    identity: str
    agents_summary: str = ""
    skills_summary: str = ""
    memory_index_summary: str = ""
    core_memory: str = ""


def build_frozen_system(sections: FrozenSystemSections) -> str:
    parts = [sections.identity]
    for field_name, label in _SECTION_ORDER:
        content = getattr(sections, field_name)
        if content:
            parts.append(f"## {label}\n\n{content}")
    return "\n\n".join(parts)
