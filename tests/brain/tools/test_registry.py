"""ToolRegistry 的单元测试：验证 evaluate/execute 两阶段的委派、异常映射与先读后改联动。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miku_on_desk.brain.providers.base import ToolDefinition, ToolUseBlock
from miku_on_desk.brain.tools.path_sandbox import PathSandbox
from miku_on_desk.brain.tools.policy import Decision, PolicyEngine, ToolPolicySpec
from miku_on_desk.brain.tools.read_tracker import ReadTracker
from miku_on_desk.brain.tools.registry import (
    ToolExecutionError,
    ToolRegistration,
    ToolRegistry,
)

_SESSION = "session-1"


def _make_registry(tmp_path: Path) -> tuple[ToolRegistry, ReadTracker]:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    sandbox = PathSandbox(cwd=cwd, output_dir=tmp_path / "output", data_dir=tmp_path / "data")
    read_tracker = ReadTracker()
    policy = PolicyEngine(
        trusted_mode=True,
        allowed_tools=frozenset(),
        denied_tools=frozenset(),
        default_decision=Decision.ASK,
        path_sandbox=sandbox,
        read_tracker=read_tracker,
    )
    return ToolRegistry(policy, read_tracker), read_tracker


async def _ok_handler(_input: dict[str, Any]) -> str:
    return "ok"


async def _raising_handler(_input: dict[str, Any]) -> str:
    raise ToolExecutionError("已知的失败原因")


async def _crashing_handler(_input: dict[str, Any]) -> str:
    raise RuntimeError("boom")


def test_definitions_returns_registered_tool_definitions(tmp_path: Path) -> None:
    registry, _ = _make_registry(tmp_path)
    definition = ToolDefinition(name="do_thing", description="d", input_schema={})
    registry.register(ToolRegistration(definition=definition, handler=_ok_handler))
    assert registry.definitions() == [definition]


def test_evaluate_denies_unknown_tool(tmp_path: Path) -> None:
    registry, _ = _make_registry(tmp_path)
    tool_use = ToolUseBlock(id="1", name="unknown", input={})
    decision = registry.evaluate(tool_use, session_id=_SESSION)
    assert decision.decision == Decision.DENY


def test_evaluate_delegates_to_policy_for_known_tool(tmp_path: Path) -> None:
    # 特意用 trusted_mode=False 构造独立的 policy，因为 _make_registry 的
    # trusted_mode=True 会按设计豁免 requires_confirmation（见 test_policy.py 里的
    # test_trusted_mode_bypasses_requires_confirmation），无法验证这里要测的委派关系。
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    sandbox = PathSandbox(cwd=cwd, output_dir=tmp_path / "output", data_dir=tmp_path / "data")
    read_tracker = ReadTracker()
    policy = PolicyEngine(
        trusted_mode=False,
        allowed_tools=frozenset(),
        denied_tools=frozenset(),
        default_decision=Decision.ASK,
        path_sandbox=sandbox,
        read_tracker=read_tracker,
    )
    registry = ToolRegistry(policy, read_tracker)
    definition = ToolDefinition(name="do_thing", description="d", input_schema={})
    registry.register(
        ToolRegistration(
            definition=definition,
            handler=_ok_handler,
            policy_spec=ToolPolicySpec(requires_confirmation=True),
        )
    )
    tool_use = ToolUseBlock(id="1", name="do_thing", input={})
    decision = registry.evaluate(tool_use, session_id=_SESSION)
    assert decision.decision == Decision.ASK


async def test_execute_returns_error_result_for_unknown_tool(tmp_path: Path) -> None:
    registry, _ = _make_registry(tmp_path)
    tool_use = ToolUseBlock(id="1", name="unknown", input={})
    result = await registry.execute(tool_use, session_id=_SESSION)
    assert result.is_error is True
    assert result.tool_use_id == "1"


async def test_execute_returns_handler_content_on_success(tmp_path: Path) -> None:
    registry, _ = _make_registry(tmp_path)
    definition = ToolDefinition(name="do_thing", description="d", input_schema={})
    registry.register(ToolRegistration(definition=definition, handler=_ok_handler))
    tool_use = ToolUseBlock(id="1", name="do_thing", input={})
    result = await registry.execute(tool_use, session_id=_SESSION)
    assert result.is_error is False
    assert result.content == "ok"


async def test_execute_maps_tool_execution_error_to_error_result(tmp_path: Path) -> None:
    registry, _ = _make_registry(tmp_path)
    definition = ToolDefinition(name="do_thing", description="d", input_schema={})
    registry.register(ToolRegistration(definition=definition, handler=_raising_handler))
    tool_use = ToolUseBlock(id="1", name="do_thing", input={})
    result = await registry.execute(tool_use, session_id=_SESSION)
    assert result.is_error is True
    assert result.content == "已知的失败原因"


async def test_execute_maps_unexpected_exception_to_generic_error_result(tmp_path: Path) -> None:
    registry, _ = _make_registry(tmp_path)
    definition = ToolDefinition(name="do_thing", description="d", input_schema={})
    registry.register(ToolRegistration(definition=definition, handler=_crashing_handler))
    tool_use = ToolUseBlock(id="1", name="do_thing", input={})
    result = await registry.execute(tool_use, session_id=_SESSION)
    assert result.is_error is True
    assert "内部错误" in result.content


async def test_execute_marks_read_tracker_when_configured(tmp_path: Path) -> None:
    registry, read_tracker = _make_registry(tmp_path)
    target = tmp_path / "cwd" / "a.txt"
    definition = ToolDefinition(name="read_file", description="d", input_schema={})
    registry.register(
        ToolRegistration(
            definition=definition,
            handler=_ok_handler,
            policy_spec=ToolPolicySpec(path_arg="path"),
            marks_read=True,
        )
    )
    tool_use = ToolUseBlock(id="1", name="read_file", input={"path": str(target)})
    await registry.execute(tool_use, session_id=_SESSION)
    assert read_tracker.has_been_read(_SESSION, target) is True


async def test_execute_does_not_mark_read_when_marks_read_is_false(tmp_path: Path) -> None:
    registry, read_tracker = _make_registry(tmp_path)
    target = tmp_path / "cwd" / "a.txt"
    definition = ToolDefinition(name="write_file", description="d", input_schema={})
    registry.register(
        ToolRegistration(
            definition=definition,
            handler=_ok_handler,
            policy_spec=ToolPolicySpec(path_arg="path", is_write=True),
        )
    )
    tool_use = ToolUseBlock(id="1", name="write_file", input={"path": str(target)})
    await registry.execute(tool_use, session_id=_SESSION)
    assert read_tracker.has_been_read(_SESSION, target) is False
