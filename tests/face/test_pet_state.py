"""PetStateMachine 的回归测试：baseline/transient/dragging 优先级合成、纯 `t` 函数语义。"""

from __future__ import annotations

from miku_on_desk.face.pet_state import PetState, PetStateMachine


def test_default_state_is_idle_at_t_zero() -> None:
    machine = PetStateMachine()
    assert machine.current_state(0.0) == PetState.IDLE


def test_set_baseline_state_changes_current_state() -> None:
    machine = PetStateMachine()
    machine.set_baseline_state(PetState.TALKING, t=1.0)
    assert machine.current_state(1.0) == PetState.TALKING
    assert machine.current_state(5.0) == PetState.TALKING


def test_set_baseline_state_with_same_state_does_not_reset_entered_at() -> None:
    machine = PetStateMachine()
    machine.set_baseline_state(PetState.TALKING, t=1.0)
    machine.set_baseline_state(PetState.TALKING, t=3.0)
    assert machine.state_entered_at(5.0) == 1.0


def test_transient_overrides_baseline_until_duration_elapses() -> None:
    machine = PetStateMachine()
    machine.set_baseline_state(PetState.IDLE, t=0.0)
    machine.trigger_transient(PetState.SUCCESS, t=1.0, duration=0.5)

    assert machine.current_state(1.0) == PetState.SUCCESS
    assert machine.current_state(1.4) == PetState.SUCCESS
    assert machine.current_state(1.5) == PetState.IDLE


def test_new_transient_overwrites_old_transient_without_queueing() -> None:
    machine = PetStateMachine()
    machine.trigger_transient(PetState.SUCCESS, t=0.0, duration=5.0)
    machine.trigger_transient(PetState.ERROR, t=0.1, duration=5.0)

    assert machine.current_state(0.2) == PetState.ERROR
    assert machine.current_state(5.2) == PetState.IDLE


def test_trigger_transient_does_not_mutate_baseline() -> None:
    machine = PetStateMachine()
    machine.set_baseline_state(PetState.TALKING, t=0.0)
    machine.trigger_transient(PetState.SUCCESS, t=1.0, duration=0.5)

    assert machine.current_state(1.6) == PetState.TALKING


def test_active_transient_still_overrides_dragging() -> None:
    """插件计划规定的优先级是 transient > dragging > baseline：拖拽途中若触发一次性
    反应（例如松手前先点了一下）,应短暂盖过 DRAGGED 显示,而不是被拖拽状态吞掉。
    """
    machine = PetStateMachine()
    machine.set_baseline_state(PetState.TALKING, t=0.0)
    machine.set_dragging(True, t=1.0)

    assert machine.current_state(1.5) == PetState.DRAGGED

    machine.trigger_transient(PetState.CLICKED, t=2.0, duration=0.5)
    assert machine.current_state(2.2) == PetState.CLICKED
    assert machine.current_state(2.6) == PetState.DRAGGED


def test_dragging_ends_and_falls_back_to_baseline() -> None:
    machine = PetStateMachine()
    machine.set_baseline_state(PetState.THINKING, t=0.0)
    machine.set_dragging(True, t=1.0)
    machine.set_dragging(False, t=2.0)

    assert machine.current_state(2.1) == PetState.THINKING


def test_state_entered_at_tracks_dragging_start() -> None:
    machine = PetStateMachine()
    machine.set_dragging(True, t=3.0)
    assert machine.state_entered_at(3.5) == 3.0


def test_default_transient_duration_used_when_state_has_no_custom_duration() -> None:
    machine = PetStateMachine()
    machine.trigger_transient(PetState.NOTICE, t=0.0)
    assert machine.current_state(0.1) == PetState.NOTICE
    assert machine.current_state(100.0) == PetState.IDLE


def test_custom_transient_durations_override_defaults() -> None:
    machine = PetStateMachine(transient_durations={PetState.SUCCESS: 10.0})
    machine.trigger_transient(PetState.SUCCESS, t=0.0)
    assert machine.current_state(5.0) == PetState.SUCCESS
    assert machine.current_state(10.5) == PetState.IDLE
