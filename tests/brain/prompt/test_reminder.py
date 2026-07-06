"""build_system_reminder / detect_response_language_hint 的行为回归测试。"""

from __future__ import annotations

from datetime import datetime

from miku_on_desk.brain.memory.models import RetrievedMemoryHint
from miku_on_desk.brain.prompt.reminder import (
    StepProgress,
    build_system_reminder,
    detect_response_language_hint,
    host_shell_descriptor,
)


def test_detect_response_language_hint_japanese() -> None:
    hint = detect_response_language_hint("こんにちは、元気ですか")
    assert hint is not None
    assert "日语" in hint


def test_detect_response_language_hint_chinese() -> None:
    hint = detect_response_language_hint("你好，今天天气怎么样")
    assert hint is not None
    assert "中文" in hint


def test_detect_response_language_hint_english() -> None:
    hint = detect_response_language_hint("Hello, how are you today")
    assert hint is not None
    assert "English" in hint


def test_detect_response_language_hint_returns_none_when_no_letters() -> None:
    assert detect_response_language_hint("1234567890") is None


def test_build_system_reminder_wraps_in_system_reminder_tag() -> None:
    reminder = build_system_reminder(
        now=datetime(2026, 7, 3, 12, 0, 0),
        latest_user_text="1234",
        host_shell="zsh on darwin",
        trusted_mode=True,
    )
    assert reminder.startswith("<system-reminder>\n")
    assert reminder.endswith("\n</system-reminder>")
    assert "2026-07-03T12:00:00" in reminder
    assert "zsh on darwin" in reminder


def test_build_system_reminder_reports_trusted_mode_state() -> None:
    trusted = build_system_reminder(
        now=datetime.now(), latest_user_text="", host_shell="sh", trusted_mode=True
    )
    untrusted = build_system_reminder(
        now=datetime.now(), latest_user_text="", host_shell="sh", trusted_mode=False
    )
    assert "已开启" in trusted
    assert "未开启" in untrusted


def test_build_system_reminder_includes_language_hint_when_detectable() -> None:
    reminder = build_system_reminder(
        now=datetime.now(), latest_user_text="你好", host_shell="sh", trusted_mode=True
    )
    assert "中文" in reminder


def test_build_system_reminder_omits_stuck_warning_when_no_progress_given() -> None:
    reminder = build_system_reminder(
        now=datetime.now(), latest_user_text="", host_shell="sh", trusted_mode=True
    )
    assert "停下来诊断" not in reminder


def test_build_system_reminder_omits_stuck_warning_when_progress_not_stuck() -> None:
    reminder = build_system_reminder(
        now=datetime.now(),
        latest_user_text="",
        host_shell="sh",
        trusted_mode=True,
        step_progress=StepProgress(step_id="step-1", attempt_count=1, elapsed_seconds=10.0),
    )
    assert "停下来诊断" not in reminder


def test_build_system_reminder_includes_stuck_warning_when_attempts_exceed_threshold() -> None:
    reminder = build_system_reminder(
        now=datetime.now(),
        latest_user_text="",
        host_shell="sh",
        trusted_mode=True,
        step_progress=StepProgress(step_id="step-1", attempt_count=3, elapsed_seconds=1.0),
    )
    assert "停下来诊断" in reminder
    assert "step-1" in reminder


def test_build_system_reminder_includes_stuck_warning_when_elapsed_exceeds_threshold() -> None:
    reminder = build_system_reminder(
        now=datetime.now(),
        latest_user_text="",
        host_shell="sh",
        trusted_mode=True,
        step_progress=StepProgress(step_id="step-2", attempt_count=1, elapsed_seconds=301.0),
    )
    assert "停下来诊断" in reminder


def test_host_shell_descriptor_returns_non_empty_string() -> None:
    assert host_shell_descriptor()


def test_build_system_reminder_omits_memory_section_when_no_relevant_memories() -> None:
    reminder = build_system_reminder(
        now=datetime.now(), latest_user_text="你好", host_shell="sh", trusted_mode=True
    )
    assert "相关记忆" not in reminder


def test_build_system_reminder_omits_memory_section_when_empty_list_given() -> None:
    reminder = build_system_reminder(
        now=datetime.now(),
        latest_user_text="你好",
        host_shell="sh",
        trusted_mode=True,
        relevant_memories=[],
    )
    assert "相关记忆" not in reminder


def test_build_system_reminder_includes_relevant_memories_when_given() -> None:
    hints = [
        RetrievedMemoryHint(label="语义", text="喜欢的饮料：美式咖啡"),
        RetrievedMemoryHint(label="情感", text="时区：Asia/Shanghai"),
    ]

    reminder = build_system_reminder(
        now=datetime.now(),
        latest_user_text="你好",
        host_shell="sh",
        trusted_mode=True,
        relevant_memories=hints,
    )

    assert "相关记忆" in reminder
    assert "[语义] 喜欢的饮料：美式咖啡" in reminder
    assert "[情感] 时区：Asia/Shanghai" in reminder


def test_build_system_reminder_memory_section_appears_after_other_lines() -> None:
    hints = [RetrievedMemoryHint(label="语义", text="k：v")]

    reminder = build_system_reminder(
        now=datetime.now(),
        latest_user_text="",
        host_shell="sh",
        trusted_mode=True,
        relevant_memories=hints,
    )

    body = reminder.removeprefix("<system-reminder>\n").removesuffix("\n</system-reminder>")
    lines = body.split("\n")
    memory_line_index = next(i for i, line in enumerate(lines) if "相关记忆" in line)
    assert memory_line_index == len(lines) - 2
    assert lines[-1] == "- [语义] k：v"
