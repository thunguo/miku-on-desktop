"""backend.py 的回归测试：find_pid_by_name 假 psutil.process_iter，open_app 假
subprocess.run——不实际枚举系统进程或启动/唤起应用。list_elements 走的是分平台
accessibility 模块，已经在 hands_eyes 的手动验证脚本里覆盖，不在这里重复。
"""

from __future__ import annotations

from typing import Any

import pytest

from miku_on_desk.brain.tools.registry import ToolExecutionError
from miku_on_desk.hands_eyes import backend as backend_module
from miku_on_desk.hands_eyes.backend import (
    ForegroundAppInfo,
    MacOSBackend,
    NullBackend,
    UIElement,
    WindowsBackend,
    create_platform_backend,
)


class _ConcreteBackend(backend_module.PlatformBackend):
    def list_elements(self, pid: int) -> list[UIElement]:
        return []

    def get_window_bounds(self, pid: int) -> tuple[int, int, int, int] | None:
        return None

    def open_app(self, name: str) -> None:
        raise NotImplementedError

    def get_idle_seconds(self) -> float:
        return 0.0

    def get_foreground_app_info(self) -> ForegroundAppInfo | None:
        return None


class _FakeProcess:
    def __init__(self, info: dict[str, Any]) -> None:
        self.info = info


def test_find_pid_by_name_matches_case_insensitively(monkeypatch: pytest.MonkeyPatch) -> None:
    processes = [
        _FakeProcess({"pid": 1, "name": "Finder"}),
        _FakeProcess({"pid": 2, "name": "Calculator"}),
    ]
    monkeypatch.setattr(backend_module.psutil, "process_iter", lambda attrs: processes)

    backend = _ConcreteBackend()
    assert backend.find_pid_by_name("calculator") == 2


def test_find_pid_by_name_strips_exe_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    processes = [_FakeProcess({"pid": 7, "name": "notepad.exe"})]
    monkeypatch.setattr(backend_module.psutil, "process_iter", lambda attrs: processes)

    backend = _ConcreteBackend()
    assert backend.find_pid_by_name("notepad") == 7


def test_find_pid_by_name_returns_none_when_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_module.psutil, "process_iter", lambda attrs: [])

    backend = _ConcreteBackend()
    assert backend.find_pid_by_name("does-not-exist") is None


def test_macos_backend_open_app_invokes_open_dash_a(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(backend_module.subprocess, "run", lambda args, check: calls.append(args))

    MacOSBackend().open_app("Calculator")

    assert calls == [["open", "-a", "Calculator"]]


def test_windows_backend_open_app_invokes_cmd_start(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(backend_module.subprocess, "run", lambda args, check: calls.append(args))

    WindowsBackend().open_app("notepad")

    assert calls == [["cmd", "/c", "start", "", "notepad"]]


def test_null_backend_queries_return_empty_results_without_raising() -> None:
    backend = NullBackend()

    assert backend.list_elements(1) == []
    assert backend.get_window_bounds(1) is None
    assert backend.get_idle_seconds() == float("inf")
    assert backend.get_foreground_app_info() is None


def test_null_backend_open_app_raises_tool_execution_error() -> None:
    with pytest.raises(ToolExecutionError):
        NullBackend().open_app("计算器")


def test_create_platform_backend_falls_back_to_null_backend_on_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend_module.sys, "platform", "linux")

    assert isinstance(create_platform_backend(), NullBackend)

