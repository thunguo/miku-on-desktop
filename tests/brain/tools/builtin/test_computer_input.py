"""computer_input 工具注册的回归测试：假 PlatformBackend，不碰真实 pynput/psutil
——那部分平台绑定的行为已经在 hands_eyes 自己的测试里覆盖。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from miku_on_desk.brain.providers.base import ToolUseBlock
from miku_on_desk.brain.tools.builtin import computer_input as computer_input_module
from miku_on_desk.brain.tools.builtin.computer_input import register_computer_input_tool
from miku_on_desk.brain.tools.path_sandbox import PathSandbox
from miku_on_desk.brain.tools.policy import Decision, PolicyEngine
from miku_on_desk.brain.tools.read_tracker import ReadTracker
from miku_on_desk.brain.tools.registry import ToolRegistry
from miku_on_desk.hands_eyes.backend import ForegroundAppInfo, PlatformBackend, UIElement


class _FakeBackend(PlatformBackend):
    def __init__(self, *, pid_after_open: int | None = 4242) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._pid_after_open = pid_after_open

    def list_elements(self, pid: int) -> list[UIElement]:
        self.calls.append(("list_elements", (pid,)))
        return []

    def get_window_bounds(self, pid: int) -> tuple[int, int, int, int] | None:
        self.calls.append(("get_window_bounds", (pid,)))
        return None

    def open_app(self, name: str) -> None:
        self.calls.append(("open_app", (name,)))

    def get_idle_seconds(self) -> float:
        return 0.0

    def get_foreground_app_info(self) -> ForegroundAppInfo | None:
        return None

    def click(self, x: int, y: int) -> None:
        self.calls.append(("click", (x, y)))

    def type_text(self, text: str) -> None:
        self.calls.append(("type_text", (text,)))

    def press_keys(self, keys: Sequence[str]) -> None:
        self.calls.append(("press_keys", (tuple(keys),)))

    def find_pid_by_name(self, name: str) -> int | None:
        self.calls.append(("find_pid_by_name", (name,)))
        return self._pid_after_open


def _make_registry(tmp_path: Path, *, trusted_mode: bool = True) -> ToolRegistry:
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    sandbox = PathSandbox(cwd=cwd, output_dir=tmp_path / "output", data_dir=tmp_path / "data")
    policy = PolicyEngine(
        trusted_mode=trusted_mode,
        allowed_tools=frozenset(),
        denied_tools=frozenset(),
        default_decision=Decision.ALLOW,
        path_sandbox=sandbox,
        read_tracker=ReadTracker(),
    )
    return ToolRegistry(policy, ReadTracker())


async def test_click_delegates_to_backend(tmp_path: Path) -> None:
    backend = _FakeBackend()
    registry = _make_registry(tmp_path)
    register_computer_input_tool(backend, registry)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="computer_input", input={"action": "click", "x": 10, "y": 20}),
        session_id="s1",
    )

    assert result.is_error is False
    assert ("click", (10, 20)) in backend.calls
    assert json.loads(result.content) == {"success": True, "action": "click"}


async def test_click_missing_coordinates_is_error(tmp_path: Path) -> None:
    backend = _FakeBackend()
    registry = _make_registry(tmp_path)
    register_computer_input_tool(backend, registry)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="computer_input", input={"action": "click"}), session_id="s1"
    )

    assert result.is_error is True
    assert "x" in result.content


async def test_type_text_delegates_to_backend(tmp_path: Path) -> None:
    backend = _FakeBackend()
    registry = _make_registry(tmp_path)
    register_computer_input_tool(backend, registry)

    result = await registry.execute(
        ToolUseBlock(
            id="c1", name="computer_input", input={"action": "type_text", "text": "你好"}
        ),
        session_id="s1",
    )

    assert result.is_error is False
    assert ("type_text", ("你好",)) in backend.calls


async def test_key_press_delegates_to_backend(tmp_path: Path) -> None:
    backend = _FakeBackend()
    registry = _make_registry(tmp_path)
    register_computer_input_tool(backend, registry)

    result = await registry.execute(
        ToolUseBlock(
            id="c1", name="computer_input", input={"action": "key_press", "keys": ["ctrl", "c"]}
        ),
        session_id="s1",
    )

    assert result.is_error is False
    assert ("press_keys", (("ctrl", "c"),)) in backend.calls


async def test_key_press_missing_keys_is_error(tmp_path: Path) -> None:
    backend = _FakeBackend()
    registry = _make_registry(tmp_path)
    register_computer_input_tool(backend, registry)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="computer_input", input={"action": "key_press", "keys": []}),
        session_id="s1",
    )

    assert result.is_error is True


async def test_open_app_delegates_and_returns_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(computer_input_module, "_OPEN_APP_SETTLE_DELAY_S", 0.0)
    backend = _FakeBackend(pid_after_open=999)
    registry = _make_registry(tmp_path)
    register_computer_input_tool(backend, registry)

    result = await registry.execute(
        ToolUseBlock(
            id="c1", name="computer_input", input={"action": "open_app", "app_name": "Calculator"}
        ),
        session_id="s1",
    )

    assert result.is_error is False
    payload = json.loads(result.content)
    assert payload == {"success": True, "action": "open_app", "pid": 999}
    assert ("open_app", ("Calculator",)) in backend.calls


async def test_open_app_skips_settle_delay_when_pid_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(computer_input_module.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(computer_input_module.time, "sleep", lambda _seconds: None)
    backend = _FakeBackend(pid_after_open=None)
    registry = _make_registry(tmp_path)
    register_computer_input_tool(backend, registry)

    result = await registry.execute(
        ToolUseBlock(
            id="c1", name="computer_input", input={"action": "open_app", "app_name": "Ghost"}
        ),
        session_id="s1",
    )

    assert result.is_error is False
    payload = json.loads(result.content)
    assert payload == {"success": True, "action": "open_app", "pid": None}
    assert sleep_calls == []


async def test_invalid_action_is_rejected(tmp_path: Path) -> None:
    backend = _FakeBackend()
    registry = _make_registry(tmp_path)
    register_computer_input_tool(backend, registry)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="computer_input", input={"action": "does-not-exist"}),
        session_id="s1",
    )

    assert result.is_error is True


def test_evaluate_requires_confirmation_when_not_trusted(tmp_path: Path) -> None:
    backend = _FakeBackend()
    registry = _make_registry(tmp_path, trusted_mode=False)
    register_computer_input_tool(backend, registry)

    decision = registry.evaluate(
        ToolUseBlock(id="c1", name="computer_input", input={"action": "click", "x": 1, "y": 1}),
        session_id="s1",
    )

    assert decision.decision is Decision.ASK


def test_evaluate_allows_in_trusted_mode(tmp_path: Path) -> None:
    backend = _FakeBackend()
    registry = _make_registry(tmp_path, trusted_mode=True)
    register_computer_input_tool(backend, registry)

    decision = registry.evaluate(
        ToolUseBlock(id="c1", name="computer_input", input={"action": "click", "x": 1, "y": 1}),
        session_id="s1",
    )

    assert decision.decision is Decision.ALLOW
