"""桌宠离散状态机：baseline（常驻）状态与 transient（一次性）反应的优先级合成。

设计核心：transient 不修改 baseline，只在 ``current_state(t)`` 的优先级判断里临时盖过
它——状态播完后"自动回到基线"不需要任何额外记账，纯粹是 ``t - transient_started_at``
超过 duration 后不再满足优先条件、自然让位给下一优先级（拖拽中/基线）。所有方法都是
关于外部单调时钟 ``t`` 的纯函数（``current_state``/``state_entered_at``），可以在测试里
任意跳跃调用，无需真实等待。
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum


class PetState(StrEnum):
    IDLE = "idle"
    TALKING = "talking"
    THINKING = "thinking"
    TOOL_RUNNING = "tool_running"
    CONFIRMATION_PENDING = "confirmation_pending"
    DRAGGED = "dragged"
    SUCCESS = "success"
    ERROR = "error"
    CLICKED = "clicked"
    NOTICE = "notice"


_DEFAULT_TRANSIENT_DURATION = 1.2

_DEFAULT_TRANSIENT_DURATIONS: dict[PetState, float] = {
    PetState.SUCCESS: 1.2,
    PetState.ERROR: 1.6,
    PetState.CLICKED: 0.5,
    PetState.NOTICE: 1.2,
}


class PetStateMachine:
    """合成 baseline / transient / dragging 三路状态,给出某一时刻应显示的 `PetState`。"""

    def __init__(self, transient_durations: Mapping[PetState, float] | None = None) -> None:
        self._transient_durations: Mapping[PetState, float] = (
            dict(_DEFAULT_TRANSIENT_DURATIONS)
            if transient_durations is None
            else dict(transient_durations)
        )
        self._baseline_state = PetState.IDLE
        self._baseline_since = 0.0
        self._transient_state: PetState | None = None
        self._transient_since = 0.0
        self._transient_duration = 0.0
        self._dragging = False
        self._dragging_since = 0.0

    def set_baseline_state(self, state: PetState, *, t: float) -> None:
        """切换常驻基线状态；只有状态真的变化时才重置计时,避免高频重复事件
        （如连续多个 ``ContentDelta``）不断重置已播放时长。
        """
        if state == self._baseline_state:
            return
        self._baseline_state = state
        self._baseline_since = t

    def trigger_transient(
        self, state: PetState, *, t: float, duration: float | None = None
    ) -> None:
        """触发一次性反应;新 transient 直接覆盖旧 transient（后来者优先,不排队）。"""
        self._transient_state = state
        self._transient_since = t
        self._transient_duration = (
            self._transient_durations.get(state, _DEFAULT_TRANSIENT_DURATION)
            if duration is None
            else duration
        )

    def set_dragging(self, dragging: bool, *, t: float) -> None:
        if dragging == self._dragging:
            return
        self._dragging = dragging
        self._dragging_since = t

    def _transient_active(self, t: float) -> bool:
        if self._transient_state is None:
            return False
        return t - self._transient_since < self._transient_duration

    def current_state(self, t: float) -> PetState:
        """纯函数,无副作用。优先级：未过期的 transient > 拖拽中 > baseline。"""
        if self._transient_active(t):
            assert self._transient_state is not None
            return self._transient_state
        if self._dragging:
            return PetState.DRAGGED
        return self._baseline_state

    def state_entered_at(self, t: float) -> float:
        """当前 ``current_state(t)`` 对应状态段的起始时间,供渲染层计算帧内已播放时长。"""
        if self._transient_active(t):
            return self._transient_since
        if self._dragging:
            return self._dragging_since
        return self._baseline_since
