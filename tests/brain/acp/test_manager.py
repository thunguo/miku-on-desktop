"""AcpManager 与 acp_delegate 工具注册的回归测试：真实 registry/policy，`run_acp_task` 打桩
——避免每个用例都拉起真实子进程，那部分协议级别的验证已经在 `test_client.py` 里覆盖。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from miku_on_desk.brain.acp import manager as acp_manager_module
from miku_on_desk.brain.acp.client import AcpTurnResult
from miku_on_desk.brain.acp.manager import AcpManager, register_acp_delegate_tool
from miku_on_desk.brain.providers.base import ToolUseBlock
from miku_on_desk.brain.tools.path_sandbox import PathSandbox
from miku_on_desk.brain.tools.policy import Decision, PolicyEngine
from miku_on_desk.brain.tools.read_tracker import ReadTracker
from miku_on_desk.brain.tools.registry import ToolRegistry
from miku_on_desk.config.settings import AcpAgentConfig

_ENABLED = AcpAgentConfig(name="claude-code", executable="claude", args=["--acp"], enabled=True)
_DISABLED = AcpAgentConfig(name="codex", executable="codex", args=[], enabled=False)


def _make_bare_registry(tmp_path: Path) -> ToolRegistry:
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    sandbox = PathSandbox(cwd=cwd, output_dir=tmp_path / "output", data_dir=tmp_path / "data")
    policy = PolicyEngine(
        trusted_mode=True,
        allowed_tools=frozenset(),
        denied_tools=frozenset(),
        default_decision=Decision.ALLOW,
        path_sandbox=sandbox,
        read_tracker=ReadTracker(),
    )
    return ToolRegistry(policy, ReadTracker())


def test_resolve_returns_none_for_unknown_agent() -> None:
    manager = AcpManager([_ENABLED])
    assert manager.resolve("does-not-exist") is None


def test_resolve_returns_none_for_disabled_agent() -> None:
    manager = AcpManager([_DISABLED])
    assert manager.resolve("codex") is None


def test_resolve_returns_config_for_enabled_agent() -> None:
    manager = AcpManager([_ENABLED])
    assert manager.resolve("claude-code") is _ENABLED


def test_register_skips_when_no_enabled_agents(tmp_path: Path) -> None:
    registry = _make_bare_registry(tmp_path)
    register_acp_delegate_tool(AcpManager([_DISABLED]), registry)

    names = [d.name for d in registry.definitions()]
    assert "acp_delegate" not in names


def test_register_exposes_tool_with_enabled_agent_names(tmp_path: Path) -> None:
    registry = _make_bare_registry(tmp_path)
    register_acp_delegate_tool(AcpManager([_ENABLED, _DISABLED]), registry)

    definitions = {d.name: d for d in registry.definitions()}
    assert "acp_delegate" in definitions
    agent_schema = definitions["acp_delegate"].input_schema["properties"]["agent"]
    assert agent_schema["enum"] == ["claude-code"]


async def test_execute_returns_success_payload_from_run_acp_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _make_bare_registry(tmp_path)
    register_acp_delegate_tool(AcpManager([_ENABLED]), registry)

    async def _fake_run_acp_task(**kwargs: object) -> AcpTurnResult:
        return AcpTurnResult(success=True, content="完成了", error=None, stop_reason="end_turn")

    monkeypatch.setattr(acp_manager_module, "run_acp_task", _fake_run_acp_task)

    result = await registry.execute(
        ToolUseBlock(
            id="call1",
            name="acp_delegate",
            input={"agent": "claude-code", "task": "写个测试", "cwd": str(tmp_path / "cwd")},
        ),
        session_id="s1",
    )

    assert result.is_error is False
    payload = json.loads(result.content)
    assert payload["success"] is True
    assert payload["content"] == "完成了"
    assert payload["stop_reason"] == "end_turn"


async def test_execute_returns_error_for_unknown_agent(tmp_path: Path) -> None:
    registry = _make_bare_registry(tmp_path)
    register_acp_delegate_tool(AcpManager([_ENABLED]), registry)

    result = await registry.execute(
        ToolUseBlock(
            id="call1",
            name="acp_delegate",
            input={"agent": "does-not-exist", "task": "x", "cwd": str(tmp_path / "cwd")},
        ),
        session_id="s1",
    )

    assert result.is_error is True
    assert "does-not-exist" in result.content


def test_evaluate_denies_cwd_outside_sandbox(tmp_path: Path) -> None:
    registry = _make_bare_registry(tmp_path)
    register_acp_delegate_tool(AcpManager([_ENABLED]), registry)

    outside = Path.home() / "miku-on-desk-test-outside-sandbox-dir-xyz"
    decision = registry.evaluate(
        ToolUseBlock(
            id="call1",
            name="acp_delegate",
            input={"agent": "claude-code", "task": "x", "cwd": str(outside)},
        ),
        session_id="s1",
    )

    assert decision.decision is Decision.DENY


def test_default_timeout_s_falls_back_to_client_default() -> None:
    manager = AcpManager([_ENABLED])
    assert manager.default_timeout_s == 900.0


def test_default_timeout_s_reflects_constructor_override() -> None:
    manager = AcpManager([_ENABLED], default_timeout_s=42.0)
    assert manager.default_timeout_s == 42.0


async def test_execute_forwards_manager_default_timeout_when_agent_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _make_bare_registry(tmp_path)
    register_acp_delegate_tool(AcpManager([_ENABLED], default_timeout_s=42.0), registry)

    captured: dict[str, object] = {}

    async def _fake_run_acp_task(**kwargs: object) -> AcpTurnResult:
        captured.update(kwargs)
        return AcpTurnResult(success=True, content="完成了", error=None, stop_reason="end_turn")

    monkeypatch.setattr(acp_manager_module, "run_acp_task", _fake_run_acp_task)

    await registry.execute(
        ToolUseBlock(
            id="call1",
            name="acp_delegate",
            input={"agent": "claude-code", "task": "写个测试", "cwd": str(tmp_path / "cwd")},
        ),
        session_id="s1",
    )

    assert captured["timeout_s"] == 42.0


async def test_execute_forwards_agent_specific_timeout_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _make_bare_registry(tmp_path)
    agent = AcpAgentConfig(
        name="claude-code", executable="claude", args=["--acp"], enabled=True, timeout_s=7.0
    )
    register_acp_delegate_tool(AcpManager([agent], default_timeout_s=42.0), registry)

    captured: dict[str, object] = {}

    async def _fake_run_acp_task(**kwargs: object) -> AcpTurnResult:
        captured.update(kwargs)
        return AcpTurnResult(success=True, content="完成了", error=None, stop_reason="end_turn")

    monkeypatch.setattr(acp_manager_module, "run_acp_task", _fake_run_acp_task)

    await registry.execute(
        ToolUseBlock(
            id="call1",
            name="acp_delegate",
            input={"agent": "claude-code", "task": "写个测试", "cwd": str(tmp_path / "cwd")},
        ),
        session_id="s1",
    )

    assert captured["timeout_s"] == 7.0


async def test_execute_forwards_path_sandbox_to_run_acp_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _make_bare_registry(tmp_path)
    sandbox = PathSandbox(
        cwd=tmp_path / "cwd", output_dir=tmp_path / "output", data_dir=tmp_path / "data"
    )
    register_acp_delegate_tool(AcpManager([_ENABLED]), registry, path_sandbox=sandbox)

    captured: dict[str, object] = {}

    async def _fake_run_acp_task(**kwargs: object) -> AcpTurnResult:
        captured.update(kwargs)
        return AcpTurnResult(success=True, content="完成了", error=None, stop_reason="end_turn")

    monkeypatch.setattr(acp_manager_module, "run_acp_task", _fake_run_acp_task)

    await registry.execute(
        ToolUseBlock(
            id="call1",
            name="acp_delegate",
            input={"agent": "claude-code", "task": "写个测试", "cwd": str(tmp_path / "cwd")},
        ),
        session_id="s1",
    )

    assert captured["path_sandbox"] is sandbox
