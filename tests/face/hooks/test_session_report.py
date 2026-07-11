"""session_report.py 的回归测试：``SessionTracker`` 的会话边界识别、战报文案、
温和成长曲线的心情平滑与里程碑文案、``GrowthStore`` 的原子读写。
"""

from __future__ import annotations

import json
from pathlib import Path

from miku_on_desk.face.hooks.schema import HookEvent
from miku_on_desk.face.hooks.session_report import (
    CompanionGrowth,
    GrowthStore,
    SessionReport,
    SessionTracker,
    format_session_report,
    growth_flavor_text,
    update_growth,
)


def _event(name: str, *, source: str = "claude_code") -> HookEvent:
    return HookEvent(event=name, source=source)


def test_session_tracker_ignores_tool_events_before_any_session_start() -> None:
    tracker = SessionTracker()

    assert tracker.observe(_event("PostToolUse"), t=1.0) is None
    assert tracker.observe(_event("SessionEnd"), t=2.0) is None


def test_session_tracker_reports_on_session_end() -> None:
    tracker = SessionTracker()
    tracker.observe(_event("SessionStart", source="claude_code"), t=0.0)
    tracker.observe(_event("PostToolUse"), t=1.0)
    tracker.observe(_event("PostToolUse"), t=2.0)
    tracker.observe(_event("PostToolUseFailure"), t=3.0)

    report = tracker.observe(_event("SessionEnd"), t=90.0)

    assert report == SessionReport(
        duration_seconds=90.0, tool_calls=3, tool_failures=1, source="claude_code"
    )


def test_session_tracker_resets_after_reporting_so_next_session_starts_clean() -> None:
    tracker = SessionTracker()
    tracker.observe(_event("SessionStart"), t=0.0)
    tracker.observe(_event("PostToolUse"), t=1.0)
    tracker.observe(_event("SessionEnd"), t=10.0)

    tracker.observe(_event("SessionStart"), t=20.0)
    report = tracker.observe(_event("SessionEnd"), t=25.0)

    assert report == SessionReport(
        duration_seconds=5.0, tool_calls=0, tool_failures=0, source="claude_code"
    )


def test_session_tracker_finalizes_previous_session_on_next_session_start() -> None:
    """Codex CLI 目前没有被安装 SessionEnd，靠下一次 SessionStart 补记上一个会话。"""
    tracker = SessionTracker()
    tracker.observe(_event("SessionStart", source="codex"), t=0.0)
    tracker.observe(_event("PostToolUse"), t=1.0)

    report = tracker.observe(_event("SessionStart", source="codex"), t=30.0)

    assert report == SessionReport(
        duration_seconds=30.0, tool_calls=1, tool_failures=0, source="codex"
    )


def test_session_tracker_first_session_start_ever_produces_no_report() -> None:
    tracker = SessionTracker()

    report = tracker.observe(_event("SessionStart"), t=0.0)

    assert report is None


def test_session_tracker_stray_session_end_without_start_is_ignored() -> None:
    tracker = SessionTracker()

    assert tracker.observe(_event("SessionEnd"), t=5.0) is None


def test_session_tracker_counts_gemini_after_tool_as_tool_call() -> None:
    tracker = SessionTracker()
    tracker.observe(_event("SessionStart", source="gemini_cli"), t=0.0)
    tracker.observe(_event("AfterTool"), t=1.0)

    report = tracker.observe(_event("SessionEnd"), t=2.0)

    assert report is not None
    assert report.tool_calls == 1
    assert report.tool_failures == 0


def test_format_session_report_zero_tool_calls_mentions_no_tool_use() -> None:
    report = SessionReport(
        duration_seconds=30.0, tool_calls=0, tool_failures=0, source="claude_code"
    )

    text = format_session_report(report)

    assert "不到一分钟" in text
    assert "次工具" not in text


def test_format_session_report_all_success_reads_smoothly() -> None:
    report = SessionReport(
        duration_seconds=600.0, tool_calls=5, tool_failures=0, source="claude_code"
    )

    text = format_session_report(report)

    assert "10 分钟" in text
    assert "5 次工具" in text
    assert "顺利" in text


def test_format_session_report_with_failures_acknowledges_hiccups_without_blame() -> None:
    report = SessionReport(
        duration_seconds=120.0, tool_calls=4, tool_failures=2, source="claude_code"
    )

    text = format_session_report(report)

    assert "2 次小状况" in text
    assert "扛过去" in text


def test_update_growth_all_success_session_pushes_mood_up() -> None:
    growth = CompanionGrowth(sessions_completed=2, mood=0.0)
    report = SessionReport(
        duration_seconds=60.0, tool_calls=4, tool_failures=0, source="claude_code"
    )

    updated = update_growth(growth, report)

    assert updated.sessions_completed == 3
    assert updated.mood > 0.0


def test_update_growth_all_failure_session_pushes_mood_down() -> None:
    growth = CompanionGrowth(sessions_completed=2, mood=0.0)
    report = SessionReport(
        duration_seconds=60.0, tool_calls=4, tool_failures=4, source="claude_code"
    )

    updated = update_growth(growth, report)

    assert updated.mood < 0.0


def test_update_growth_single_session_cannot_swing_mood_to_extreme() -> None:
    """ "温和"成长曲线的核心断言：旧心情占大头，单次会话不该把心情直接打到 -1/1。"""
    growth = CompanionGrowth(sessions_completed=0, mood=0.0)
    report = SessionReport(
        duration_seconds=60.0, tool_calls=1, tool_failures=1, source="claude_code"
    )

    updated = update_growth(growth, report)

    assert -1.0 < updated.mood < 1.0
    assert abs(updated.mood) <= 0.3 + 1e-9


def test_update_growth_zero_tool_calls_session_is_neutral() -> None:
    growth = CompanionGrowth(sessions_completed=0, mood=0.4)
    report = SessionReport(
        duration_seconds=60.0, tool_calls=0, tool_failures=0, source="claude_code"
    )

    updated = update_growth(growth, report)

    assert updated.mood == growth.mood * 0.7


def test_growth_flavor_text_milestone_session_takes_priority_over_mood() -> None:
    growth = CompanionGrowth(sessions_completed=10, mood=-0.9)

    text = growth_flavor_text(growth)

    assert text is not None
    assert "第 10 次" in text


def test_growth_flavor_text_high_mood_without_milestone() -> None:
    growth = CompanionGrowth(sessions_completed=3, mood=0.8)

    text = growth_flavor_text(growth)

    assert text == "感觉最近的状态很不错呢！"


def test_growth_flavor_text_low_mood_is_reassuring_not_guilt_tripping() -> None:
    growth = CompanionGrowth(sessions_completed=3, mood=-0.8)

    text = growth_flavor_text(growth)

    assert text is not None
    assert "一直都在" in text
    assert "回来" not in text
    assert "记得" not in text


def test_growth_flavor_text_neutral_mood_returns_none() -> None:
    growth = CompanionGrowth(sessions_completed=3, mood=0.1)

    assert growth_flavor_text(growth) is None


def test_growth_store_load_missing_file_returns_default(tmp_path: Path) -> None:
    store = GrowthStore(tmp_path / "companion_growth.json")

    assert store.load() == CompanionGrowth()


def test_growth_store_save_then_load_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "companion_growth.json"
    store = GrowthStore(path)
    growth = CompanionGrowth(sessions_completed=7, mood=0.42)

    store.save(growth)

    assert store.load() == growth


def test_growth_store_load_recovers_from_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "companion_growth.json"
    path.write_text("not json", encoding="utf-8")
    store = GrowthStore(path)

    assert store.load() == CompanionGrowth()


def test_growth_store_save_overwrites_without_leaving_tmp_files(tmp_path: Path) -> None:
    path = tmp_path / "companion_growth.json"
    store = GrowthStore(path)

    store.save(CompanionGrowth(sessions_completed=1, mood=0.1))
    store.save(CompanionGrowth(sessions_completed=2, mood=0.2))

    assert json.loads(path.read_text(encoding="utf-8"))["sessions_completed"] == 2
    assert list(path.parent.glob("*.tmp-*")) == []
