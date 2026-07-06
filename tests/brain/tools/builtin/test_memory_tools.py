"""remember/recall 工具注册的回归测试：验证参数校验、写入落盘、search 命中、不需要确认。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from miku_on_desk.brain.memory.models import Fact
from miku_on_desk.brain.memory.system import MemorySystem
from miku_on_desk.brain.providers.base import ToolUseBlock
from miku_on_desk.brain.tools.builtin.memory_tools import register_memory_tools
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
    return ToolRegistry(policy, ReadTracker())


@pytest.fixture
def system(tmp_path: Path) -> MemorySystem:
    return MemorySystem(tmp_path / "memory")


async def test_remember_writes_active_fact_with_tool_source(
    tmp_path: Path, system: MemorySystem
) -> None:
    registry = _make_registry(tmp_path)
    register_memory_tools(system, registry)

    result = await registry.execute(
        ToolUseBlock(
            id="c1",
            name="remember",
            input={"key": "habits/sleep_schedule", "value": "喜欢熬夜到凌晨两点"},
        ),
        session_id="s1",
    )

    assert result.is_error is False
    facts = system.semantic.list_facts(subject="user", status="active")
    matches = [fact for fact in facts if fact.predicate == "habits/sleep_schedule"]
    assert len(matches) == 1
    assert matches[0].object == "喜欢熬夜到凌晨两点"
    assert matches[0].extracted_by == "tool:remember"


async def test_remember_same_key_overwrites_previous_value(
    tmp_path: Path, system: MemorySystem
) -> None:
    registry = _make_registry(tmp_path)
    register_memory_tools(system, registry)

    await registry.execute(
        ToolUseBlock(id="c1", name="remember", input={"key": "habits/coffee", "value": "喝美式"}),
        session_id="s1",
    )
    await registry.execute(
        ToolUseBlock(id="c2", name="remember", input={"key": "habits/coffee", "value": "喝拿铁"}),
        session_id="s1",
    )

    facts = system.semantic.list_facts(subject="user", status="active")
    matches = [fact for fact in facts if fact.predicate == "habits/coffee"]
    assert len(matches) == 1
    assert matches[0].object == "喝拿铁"


async def test_remember_rejects_missing_fields(tmp_path: Path, system: MemorySystem) -> None:
    registry = _make_registry(tmp_path)
    register_memory_tools(system, registry)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="remember", input={"key": "habits/coffee"}),
        session_id="s1",
    )

    assert result.is_error is True


async def test_recall_finds_remembered_fact_by_key_or_value_substring(
    tmp_path: Path, system: MemorySystem
) -> None:
    registry = _make_registry(tmp_path)
    register_memory_tools(system, registry)
    now = datetime.now(UTC).isoformat()
    system.semantic.upsert_fact(
        Fact(
            id="",
            subject="user",
            subject_type="person",
            predicate="speech_style/short_sentences",
            object="说话喜欢用短句",
            object_type="concept",
            confidence=1.0,
            source=[],
            valid_from=now,
            recorded_at=now,
            extracted_by="llm:fast",
            status="active",
        )
    )

    result = await registry.execute(
        ToolUseBlock(id="c1", name="recall", input={"query": "短句"}),
        session_id="s1",
    )

    assert result.is_error is False
    assert "speech_style/short_sentences" in result.content


async def test_recall_rejects_missing_query(tmp_path: Path, system: MemorySystem) -> None:
    registry = _make_registry(tmp_path)
    register_memory_tools(system, registry)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="recall", input={}),
        session_id="s1",
    )

    assert result.is_error is True


@pytest.mark.parametrize("tool_name", ["remember", "recall"])
def test_evaluate_allows_when_not_trusted(
    tmp_path: Path, system: MemorySystem, tool_name: str
) -> None:
    registry = _make_registry(tmp_path, trusted_mode=False)
    register_memory_tools(system, registry)

    decision = registry.evaluate(
        ToolUseBlock(id="c1", name=tool_name, input={"key": "k", "value": "v", "query": "q"}),
        session_id="s1",
    )

    assert decision.decision is Decision.ALLOW
