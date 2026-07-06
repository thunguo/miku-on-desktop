"""schema.py 的回归测试：事件名→状态转移查表、``HookEvent.from_raw`` 的字段提取与默认值。"""

from __future__ import annotations

from miku_on_desk.face.hooks.schema import (
    HookEvent,
    Transition,
    TransitionKind,
    resolve_transition,
)
from miku_on_desk.face.pet_state import PetState


def test_resolve_transition_known_event_returns_mapped_transition() -> None:
    assert resolve_transition("SessionStart") == Transition(TransitionKind.BASELINE, PetState.IDLE)
    assert resolve_transition("PostToolUseFailure") == Transition(
        TransitionKind.TRANSIENT, PetState.ERROR
    )


def test_resolve_transition_unknown_event_returns_none() -> None:
    assert resolve_transition("SomeFutureEventNobodyHeardOf") is None


def test_hook_event_from_raw_extracts_all_known_fields() -> None:
    raw = {
        "event": "PostToolUseFailure",
        "tool_name": "Bash",
        "error": "command not found: foo",
        "reason": "boom",
        "source": "claude_code",
    }

    event = HookEvent.from_raw(raw)

    assert event.event == "PostToolUseFailure"
    assert event.tool_name == "Bash"
    assert event.error == "command not found: foo"
    assert event.reason == "boom"
    assert event.source == "claude_code"
    assert event.raw == raw


def test_hook_event_from_raw_prefers_event_field_over_hook_event_name() -> None:
    raw = {"event": "PostToolUse", "hook_event_name": "PreToolUse"}

    event = HookEvent.from_raw(raw)

    assert event.event == "PostToolUse"


def test_hook_event_from_raw_falls_back_to_hook_event_name_field() -> None:
    raw = {"hook_event_name": "SessionStart"}

    event = HookEvent.from_raw(raw)

    assert event.event == "SessionStart"


def test_hook_event_from_raw_missing_event_name_defaults_to_empty_string() -> None:
    event = HookEvent.from_raw({})

    assert event.event == ""


def test_hook_event_from_raw_missing_optional_fields_default_to_none() -> None:
    event = HookEvent.from_raw({"event": "SessionStart"})

    assert event.tool_name is None
    assert event.error is None
    assert event.reason is None


def test_hook_event_from_raw_missing_source_defaults_to_claude_code() -> None:
    event = HookEvent.from_raw({"event": "SessionStart"})

    assert event.source == "claude_code"


def test_hook_event_direct_construction_missing_source_defaults_to_unknown() -> None:
    event = HookEvent(event="SessionStart")

    assert event.source == "unknown"


def test_hook_event_from_raw_preserves_extra_unrecognized_keys_in_raw() -> None:
    raw = {"event": "SessionStart", "some_future_field": "value"}

    event = HookEvent.from_raw(raw)

    assert event.raw == raw
