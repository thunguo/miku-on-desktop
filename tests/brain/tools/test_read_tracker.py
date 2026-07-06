"""ReadTracker 的单元测试：验证先读后改追踪按 session 隔离，且能显式清空。"""

from __future__ import annotations

from pathlib import Path

from miku_on_desk.brain.tools.read_tracker import ReadTracker


def test_has_not_been_read_initially(tmp_path: Path) -> None:
    tracker = ReadTracker()
    assert tracker.has_been_read("session-1", tmp_path / "a.txt") is False


def test_mark_read_makes_has_been_read_true(tmp_path: Path) -> None:
    tracker = ReadTracker()
    target = tmp_path / "a.txt"
    tracker.mark_read("session-1", target)
    assert tracker.has_been_read("session-1", target) is True


def test_read_state_is_scoped_by_session(tmp_path: Path) -> None:
    tracker = ReadTracker()
    target = tmp_path / "a.txt"
    tracker.mark_read("session-1", target)
    assert tracker.has_been_read("session-2", target) is False


def test_clear_session_removes_all_read_state(tmp_path: Path) -> None:
    tracker = ReadTracker()
    target = tmp_path / "a.txt"
    tracker.mark_read("session-1", target)
    tracker.clear_session("session-1")
    assert tracker.has_been_read("session-1", target) is False
