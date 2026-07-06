"""express_reaction 工具注册的回归测试：验证参数校验、事件总线投递、不需要确认这三件事。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from miku_on_desk.brain.providers.base import ToolUseBlock
from miku_on_desk.brain.tools.builtin.express_reaction import register_express_reaction_tool
from miku_on_desk.brain.tools.path_sandbox import PathSandbox
from miku_on_desk.brain.tools.policy import Decision, PolicyEngine
from miku_on_desk.brain.tools.read_tracker import ReadTracker
from miku_on_desk.brain.tools.registry import ToolRegistry
from miku_on_desk.bridge.events import BrainEventBus, ReactionKind, ReactionTriggered


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


@pytest.mark.parametrize("kind", list(ReactionKind))
async def test_valid_kind_emits_event_and_returns_success(
    tmp_path: Path, kind: ReactionKind
) -> None:
    bus = BrainEventBus()
    captured: list[object] = []
    bus.brain_event.connect(captured.append)
    registry = _make_registry(tmp_path)
    register_express_reaction_tool(bus, registry)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="express_reaction", input={"kind": kind.value}),
        session_id="s1",
    )

    assert result.is_error is False
    assert json.loads(result.content) == {"success": True, "kind": kind.value}
    assert captured == [ReactionTriggered(kind=kind)]


async def test_invalid_kind_is_rejected(tmp_path: Path) -> None:
    bus = BrainEventBus()
    registry = _make_registry(tmp_path)
    register_express_reaction_tool(bus, registry)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="express_reaction", input={"kind": "does-not-exist"}),
        session_id="s1",
    )

    assert result.is_error is True


def test_evaluate_allows_in_trusted_mode(tmp_path: Path) -> None:
    bus = BrainEventBus()
    registry = _make_registry(tmp_path, trusted_mode=True)
    register_express_reaction_tool(bus, registry)

    decision = registry.evaluate(
        ToolUseBlock(id="c1", name="express_reaction", input={"kind": "happy"}), session_id="s1"
    )

    assert decision.decision is Decision.ALLOW


def test_evaluate_allows_when_not_trusted(tmp_path: Path) -> None:
    bus = BrainEventBus()
    registry = _make_registry(tmp_path, trusted_mode=False)
    register_express_reaction_tool(bus, registry)

    decision = registry.evaluate(
        ToolUseBlock(id="c1", name="express_reaction", input={"kind": "happy"}), session_id="s1"
    )

    assert decision.decision is Decision.ALLOW
