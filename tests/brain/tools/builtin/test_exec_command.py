"""exec_command.py 的回归测试：stdout/stderr 合并、输出截断、超时终止进程、二次确认。"""

from __future__ import annotations

from pathlib import Path

import pytest

from miku_on_desk.brain.providers.base import ToolUseBlock
from miku_on_desk.brain.tools.builtin import exec_command as exec_command_module
from miku_on_desk.brain.tools.builtin.exec_command import register_exec_command_tool
from miku_on_desk.brain.tools.path_sandbox import PathSandbox
from miku_on_desk.brain.tools.policy import Decision, PolicyEngine
from miku_on_desk.brain.tools.read_tracker import ReadTracker
from miku_on_desk.brain.tools.registry import ToolRegistry


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
    registry = ToolRegistry(policy, ReadTracker())
    register_exec_command_tool(registry)
    return registry


async def test_exec_command_returns_stdout_and_exit_code(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="exec_command", input={"command": "printf hello"}),
        session_id="s1",
    )

    assert result.is_error is False
    assert "exit_code=0" in result.content
    assert "hello" in result.content


async def test_exec_command_merges_stderr_into_stdout(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="exec_command", input={"command": "echo oops 1>&2"}),
        session_id="s1",
    )

    assert result.is_error is False
    assert "oops" in result.content


async def test_exec_command_nonzero_exit_code_is_not_a_tool_error(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="exec_command", input={"command": "exit 7"}),
        session_id="s1",
    )

    assert result.is_error is False
    assert "exit_code=7" in result.content


async def test_exec_command_truncates_large_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exec_command_module, "_MAX_OUTPUT_CHARS", 10)
    registry = _make_registry(tmp_path)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="exec_command", input={"command": "yes x | head -c 100"}),
        session_id="s1",
    )

    assert result.is_error is False
    assert "已截断" in result.content


async def test_exec_command_timeout_kills_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exec_command_module, "_TIMEOUT_S", 0.05)
    registry = _make_registry(tmp_path)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="exec_command", input={"command": "sleep 5"}),
        session_id="s1",
    )

    assert result.is_error is True
    assert "终止" in result.content


async def test_exec_command_missing_command_is_error(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="exec_command", input={}), session_id="s1"
    )

    assert result.is_error is True


def test_evaluate_requires_confirmation_when_not_trusted(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path, trusted_mode=False)

    decision = registry.evaluate(
        ToolUseBlock(id="c1", name="exec_command", input={"command": "echo hi"}),
        session_id="s1",
    )

    assert decision.decision is Decision.ASK


def test_evaluate_allows_in_trusted_mode(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path, trusted_mode=True)

    decision = registry.evaluate(
        ToolUseBlock(id="c1", name="exec_command", input={"command": "echo hi"}),
        session_id="s1",
    )

    assert decision.decision is Decision.ALLOW
