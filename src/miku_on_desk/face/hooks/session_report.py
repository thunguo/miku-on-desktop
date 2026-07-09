"""外部 CLI 编程会话的战报生成 + 温和成长曲线。

``SessionTracker`` 把一串 ``HookEvent`` 折叠成"这次编程冒险"用了多久、调用了几次工具、
踩了几次坑的统计（``SessionReport``），会话边界识别策略：

- Claude Code / Gemini CLI 都会发 ``SessionEnd``，收到即结束当前会话并出报告。
- Codex CLI 目前没有被列进 ``installer._MANAGED_EVENTS_CODEX`` 的会话结束事件，因此额外
  兼容"下一次 ``SessionStart`` 到达时，把上一个未正常收尾的会话补记出报告"这条路径——
  不为此新猜测/新增一个未经核实的 Codex 事件名。

``CompanionGrowth`` 是跨进程持久化的长期心情曲线：用指数滑动平均把"这次会话顺不顺利"
揉进一个 [-1, 1] 的心情值，刻意做得很钝——单次会话的好坏不会让心情剧烈跳动，也没有任何
连续签到/最后一次登录时间之类的字段。``growth_flavor_text`` 只在心情明显偏好/偏差,或
撞上里程碑次数时才追加一句话，其它时候返回 ``None``、不硬凑话说；措辞上不出现"记得回来"
"好久不见"这类挽留/催促语气——这是本项目对同类桌宠产品暗黑模式的刻意规避,而不是疏漏。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from miku_on_desk.face.hooks.schema import HookEvent

logger = logging.getLogger(__name__)

_SESSION_START_EVENTS = frozenset({"SessionStart"})
_SESSION_END_EVENTS = frozenset({"SessionEnd"})
_TOOL_SUCCESS_EVENTS = frozenset({"PostToolUse", "AfterTool"})
_TOOL_FAILURE_EVENTS = frozenset({"PostToolUseFailure"})


@dataclass(frozen=True)
class SessionReport:
    """一次已结束会话的统计摘要，供 ``format_session_report``/``update_growth`` 消费。"""

    duration_seconds: float
    tool_calls: int
    tool_failures: int
    source: str


@dataclass
class _SessionInProgress:
    started_at: float
    source: str
    tool_calls: int = 0
    tool_failures: int = 0


class SessionTracker:
    """累积单个进行中会话的事件，在会话边界到达时吐出一份 ``SessionReport``。"""

    def __init__(self) -> None:
        self._current: _SessionInProgress | None = None

    def observe(self, event: HookEvent, *, t: float) -> SessionReport | None:
        name = event.event
        if name in _SESSION_START_EVENTS:
            pending = self._finalize(t) if self._current is not None else None
            self._current = _SessionInProgress(started_at=t, source=event.source)
            return pending
        if name in _SESSION_END_EVENTS:
            return self._finalize(t) if self._current is not None else None
        if self._current is None:
            return None
        if name in _TOOL_SUCCESS_EVENTS:
            self._current.tool_calls += 1
        elif name in _TOOL_FAILURE_EVENTS:
            self._current.tool_calls += 1
            self._current.tool_failures += 1
        return None

    def _finalize(self, t: float) -> SessionReport:
        current = self._current
        assert current is not None
        self._current = None
        return SessionReport(
            duration_seconds=max(0.0, t - current.started_at),
            tool_calls=current.tool_calls,
            tool_failures=current.tool_failures,
            source=current.source,
        )


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return "不到一分钟"
    return f"{round(seconds / 60)} 分钟"


def format_session_report(report: SessionReport) -> str:
    """生成中文战报小结，语气温暖、不带数字压迫感——没有连续天数、没有"再接再厉"式鞭策。"""
    duration_text = _format_duration(report.duration_seconds)
    if report.tool_calls == 0:
        return f"这次一起待了 {duration_text}，虽然没看到什么工具调用，能陪着你也很好呀。"

    lines = [f"这次一起写代码大约 {duration_text}，一共用了 {report.tool_calls} 次工具。"]
    if report.tool_failures > 0:
        lines.append(f"中间遇到了 {report.tool_failures} 次小状况，不过都一起扛过去啦！")
    else:
        lines.append("一路都很顺利～")
    return "\n".join(lines)


@dataclass(frozen=True)
class CompanionGrowth:
    """跨进程持久化的长期心情曲线，刻意只有两个字段——没有任何时间戳/连续天数。"""

    sessions_completed: int = 0
    mood: float = 0.0


# 越接近 1，旧印象占比越高，单次会话的好坏对心情的影响就越钝——这是"温和"成长曲线的核心。
_MOOD_SMOOTHING = 0.7
_MOOD_FLAVOR_THRESHOLD = 0.5
_MILESTONE_SESSIONS = (1, 10, 50, 100, 200)


def _session_score(report: SessionReport) -> float:
    if report.tool_calls == 0:
        return 0.0
    failure_ratio = report.tool_failures / report.tool_calls
    return max(-1.0, min(1.0, 1.0 - 2.0 * failure_ratio))


def update_growth(growth: CompanionGrowth, report: SessionReport) -> CompanionGrowth:
    score = _session_score(report)
    mood = growth.mood * _MOOD_SMOOTHING + score * (1 - _MOOD_SMOOTHING)
    return CompanionGrowth(
        sessions_completed=growth.sessions_completed + 1,
        mood=max(-1.0, min(1.0, mood)),
    )


def growth_flavor_text(growth: CompanionGrowth) -> str | None:
    """里程碑优先于心情——次数更少见，也更值得单独说一句;否则按心情偏好/偏差各给一句,
    平淡区间（多数会话都会落在这里）返回 ``None``，不用无意义的话硬凑。
    """
    if growth.sessions_completed in _MILESTONE_SESSIONS:
        return f"这是我们第 {growth.sessions_completed} 次一起写代码，很高兴能陪着你～"
    if growth.mood >= _MOOD_FLAVOR_THRESHOLD:
        return "感觉最近的状态很不错呢！"
    if growth.mood <= -_MOOD_FLAVOR_THRESHOLD:
        return "最近好像常常碰到小麻烦，不过别担心，我一直都在。"
    return None


_GROWTH_FILE_VERSION = "1.0"


class GrowthStore:
    """``companion_growth.json`` 的原子读写外壳，写法与 ``emotional_store.py`` 一致。

    读写失败（文件损坏、磁盘不可写等）都只记日志、回退到默认值/放弃保存——这是装饰性的
    情绪反馈，不是核心功能,不应该因为这个文件的问题连带影响 hook 事件处理或应用启动。
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> CompanionGrowth:
        if not self._path.exists():
            return CompanionGrowth()
        try:
            data: dict[str, Any] = json.loads(self._path.read_text(encoding="utf-8"))
            return CompanionGrowth(
                sessions_completed=int(data.get("sessions_completed", 0)),
                mood=float(data.get("mood", 0.0)),
            )
        except Exception:
            logger.exception("读取成长曲线文件失败，回退到默认值：%s", self._path)
            return CompanionGrowth()

    def save(self, growth: CompanionGrowth) -> None:
        payload = {
            "version": _GROWTH_FILE_VERSION,
            "sessions_completed": growth.sessions_completed,
            "mood": growth.mood,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(f"{self._path.suffix}.tmp-{uuid.uuid4().hex}")
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(tmp_path, self._path)
        except Exception:
            logger.exception("保存成长曲线文件失败，跳过：%s", self._path)
