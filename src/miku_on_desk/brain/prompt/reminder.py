"""每轮临时提醒：拼一段 ``<system-reminder>`` 追加到当轮用户消息的拷贝上，从不写回历史。

范围按本项目实际拥有的子系统收窄：不含配对设备状态、议程摘要等内容——配对设备在本项目里
不存在，议程子系统对应尚未实现的任务，等它就位后再按需扩展这里的分区，现在硬塞一个空字段
没有任何测试或使用价值。当前保留五类信息：本地时间、响应语言判断、宿主 shell 画像、执行
卡住检测、与当轮用户输入相关的记忆召回。

这段文本必须只追加在"当轮用户消息的拷贝"上，绝不能经过 `frozen_system.py` 那条路径进入
系统提示：这里的内容逐轮变化，一旦流经冻结系统提示的缓存前缀，就会让每一轮都作废缓存，
抵消掉冻结系统提示本身存在的意义。这也是为什么"相关记忆"这段是从调用方已经查好的
``RetrievedMemoryHint`` 列表格式化出来的，而不是本模块自己去
`memory_system.retrieve_hints(...)`——本模块保持零 I/O、纯函数，查询本身（何时查、查多少条）
由调用方（`main.py`）决定。
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from miku_on_desk.brain.memory.models import RetrievedMemoryHint

_HIRAGANA_KATAKANA = range(0x3040, 0x30FF + 1)
_CJK_IDEOGRAPHS = range(0x4E00, 0x9FFF + 1)

_STUCK_ATTEMPT_THRESHOLD = 3
_STUCK_ELAPSED_THRESHOLD_S = 300.0


def host_shell_descriptor() -> str:
    """当前宿主 shell/平台画像，主循环每轮 reminder 与 spawn_agents 的 sub-agent system 提示
    共用同一份计算，保证两条路径对模型呈现的平台描述不会出现分歧。"""
    shell = os.environ.get("SHELL")
    if shell:
        return f"{Path(shell).name} on {sys.platform}"
    return sys.platform


@dataclass(frozen=True)
class StepProgress:
    """当前执行计划里某一步的进度；用于判断是否需要提醒模型"停下来诊断"。"""

    step_id: str
    attempt_count: int
    elapsed_seconds: float


def detect_response_language_hint(text: str) -> str | None:
    """按 Unicode 脚本粗略判断用户消息语言；无法判断时返回 ``None``（不猜、不硬凑一句废话）。"""
    has_kana = any(ord(ch) in _HIRAGANA_KATAKANA for ch in text)
    if has_kana:
        return "用户消息包含日文假名，除非用户明确要求，否则请用日语回复。"
    has_cjk = any(ord(ch) in _CJK_IDEOGRAPHS for ch in text)
    if has_cjk:
        return "用户消息主要是中文，除非用户明确要求，否则请用中文回复。"
    letters = [ch for ch in text if ch.isalpha()]
    if letters and all(ord(ch) < 128 for ch in letters):
        return "The user's message is in English; respond in English unless asked otherwise."
    return None


def _format_stuck_warning(progress: StepProgress) -> str | None:
    stuck = (
        progress.attempt_count >= _STUCK_ATTEMPT_THRESHOLD
        or progress.elapsed_seconds >= _STUCK_ELAPSED_THRESHOLD_S
    )
    if not stuck:
        return None
    return (
        f'当前步骤"{progress.step_id}"已尝试 {progress.attempt_count} 次，'
        f"耗时 {progress.elapsed_seconds:.0f} 秒——停下来诊断根因、修订计划，"
        "不要继续用同样的方式重试。"
    )


def _format_relevant_memories(hints: Sequence[RetrievedMemoryHint]) -> str | None:
    if not hints:
        return None
    lines = ["相关记忆（仅供参考，不代表用户此刻在问这些）："]
    lines.extend(f"- [{hint.label}] {hint.text}" for hint in hints)
    return "\n".join(lines)


def build_system_reminder(
    *,
    now: datetime,
    latest_user_text: str,
    host_shell: str,
    trusted_mode: bool,
    step_progress: StepProgress | None = None,
    relevant_memories: Sequence[RetrievedMemoryHint] = (),
) -> str:
    lines = [f"当前本地时间：{now.isoformat(timespec='seconds')}", f"宿主环境：{host_shell}"]
    if trusted_mode:
        lines.append("信任模式：已开启，工具调用默认放行")
    else:
        lines.append("信任模式：未开启，敏感操作需用户确认")

    language_hint = detect_response_language_hint(latest_user_text)
    if language_hint is not None:
        lines.append(language_hint)

    if step_progress is not None:
        stuck_warning = _format_stuck_warning(step_progress)
        if stuck_warning is not None:
            lines.append(stuck_warning)

    memories_section = _format_relevant_memories(relevant_memories)
    if memories_section is not None:
        lines.append(memories_section)

    body = "\n".join(lines)
    return f"<system-reminder>\n{body}\n</system-reminder>"
