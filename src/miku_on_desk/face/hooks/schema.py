"""外部 hook 事件 → 桌宠状态转移的映射决策。

传输层（``server.py``）不认识 ``PetState``，把"哪个事件名对应哪种状态转移"集中在这一处，
与 ``overlay_window.py`` 里 ``_on_brain_event`` 的写法对称。未知事件名交给调用方决定如何
处理（本项目选择记录日志后忽略、不报错），因为外部 CLI 工具的事件命名会独立于本项目的
发布节奏演进，新增/改名事件不应该让 sidecar 报错。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from pydantic import BaseModel, Field

from miku_on_desk.face.pet_state import PetState


class TransitionKind(Enum):
    BASELINE = auto()
    TRANSIENT = auto()


@dataclass(frozen=True)
class Transition:
    kind: TransitionKind
    state: PetState


_EVENT_TRANSITIONS: dict[str, Transition] = {
    "SessionStart": Transition(TransitionKind.BASELINE, PetState.IDLE),
    "UserPromptSubmit": Transition(TransitionKind.BASELINE, PetState.THINKING),
    "PreToolUse": Transition(TransitionKind.BASELINE, PetState.TOOL_RUNNING),
    "PostToolUse": Transition(TransitionKind.TRANSIENT, PetState.SUCCESS),
    "PostToolUseFailure": Transition(TransitionKind.TRANSIENT, PetState.ERROR),
    "PermissionRequest": Transition(TransitionKind.BASELINE, PetState.CONFIRMATION_PENDING),
    "PermissionDenied": Transition(TransitionKind.TRANSIENT, PetState.ERROR),
    "Stop": Transition(TransitionKind.TRANSIENT, PetState.SUCCESS),
    "StopFailure": Transition(TransitionKind.TRANSIENT, PetState.ERROR),
    "SessionEnd": Transition(TransitionKind.BASELINE, PetState.IDLE),
    # Codex CLI 专属事件（SessionStart/PreToolUse/PostToolUse/UserPromptSubmit/Stop/
    # PermissionRequest 与 Claude Code 同名，复用上面的条目，不重复列出）。
    "SubagentStart": Transition(TransitionKind.BASELINE, PetState.TOOL_RUNNING),
    "SubagentStop": Transition(TransitionKind.TRANSIENT, PetState.SUCCESS),
    "PreCompact": Transition(TransitionKind.TRANSIENT, PetState.NOTICE),
    "PostCompact": Transition(TransitionKind.TRANSIENT, PetState.NOTICE),
    # Gemini CLI 专属事件。
    "BeforeTool": Transition(TransitionKind.BASELINE, PetState.TOOL_RUNNING),
    "AfterTool": Transition(TransitionKind.TRANSIENT, PetState.SUCCESS),
    "BeforeAgent": Transition(TransitionKind.BASELINE, PetState.THINKING),
    "AfterAgent": Transition(TransitionKind.TRANSIENT, PetState.SUCCESS),
    "Notification": Transition(TransitionKind.BASELINE, PetState.CONFIRMATION_PENDING),
}


def resolve_transition(event_name: str) -> Transition | None:
    return _EVENT_TRANSITIONS.get(event_name)


def known_event_names() -> list[str]:
    return list(_EVENT_TRANSITIONS)


class HookEvent(BaseModel):
    """一次外部 hook 通知，字段是本项目自定义的规整形状，不追求与调用方原始 payload
    字节级一致——``raw`` 保留完整原始 JSON，供将来对接真实 Claude Code 时按需调整
    ``from_raw`` 的取值逻辑，而不必改这里的 schema 或下游消费代码。
    """

    event: str
    tool_name: str | None = None
    error: str | None = None
    reason: str | None = None
    source: str = "unknown"
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> HookEvent:
        """``event``是本项目自定义的字段名；同时兼容 Claude Code 官方 hook payload
        实际使用的 ``hook_event_name`` 字段——两者具体以哪个为准，需要在真实 Claude
        Code 环境里验证后再收窄。
        """
        event = raw.get("event") or raw.get("hook_event_name") or ""
        return cls(
            event=str(event),
            tool_name=raw.get("tool_name"),
            error=raw.get("error"),
            reason=raw.get("reason"),
            source=raw.get("source", "claude_code"),
            raw=raw,
        )
