"""spawn_agents 工具与 run_sub_agent 的回归测试：假 Provider 驱动，不接真实 LLM。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from miku_on_desk.brain.agents.manager import AgentManager
from miku_on_desk.brain.agents.spawn import register_spawn_agents_tool, run_sub_agent
from miku_on_desk.brain.model_router import ModelRouter
from miku_on_desk.brain.providers.base import (
    Message,
    OnContent,
    OnThinking,
    Provider,
    StreamResult,
    ToolDefinition,
    ToolUseBlock,
)
from miku_on_desk.brain.tools.path_sandbox import PathSandbox
from miku_on_desk.brain.tools.policy import Decision, PolicyEngine
from miku_on_desk.brain.tools.read_tracker import ReadTracker
from miku_on_desk.brain.tools.registry import ToolRegistration, ToolRegistry
from miku_on_desk.config.settings import ModelRouterConfig, ModelTier, ProviderConfig, ProviderName


async def _ok_handler(_input: dict[str, Any]) -> str:
    return "ok"


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


def _make_registry_with_echo_and_spawn_agents(tmp_path: Path) -> ToolRegistry:
    registry = _make_bare_registry(tmp_path)
    registry.register(
        ToolRegistration(
            definition=ToolDefinition(name="echo_tool", description="d", input_schema={}),
            handler=_ok_handler,
        )
    )
    registry.register(
        ToolRegistration(
            definition=ToolDefinition(name="spawn_agents", description="d", input_schema={}),
            handler=_ok_handler,
        )
    )
    return registry


def _router(model_id: str = "model-x") -> ModelRouter:
    config = ModelRouterConfig()
    config.anthropic = ProviderConfig(api_key="key", models={ModelTier.FAST: model_id})
    return ModelRouter(config)


def _multi_provider_router() -> ModelRouter:
    config = ModelRouterConfig()
    config.anthropic = ProviderConfig(api_key="key", models={ModelTier.FAST: "model-fast"})
    config.openai = ProviderConfig(api_key="key", models={ModelTier.MEDIUM: "model-slow"})
    return ModelRouter(config)


@pytest.fixture
def agent_manager(tmp_path: Path) -> AgentManager:
    return AgentManager(tmp_path / "agents.db")


class _FakeProvider(Provider):
    def __init__(self, results: list[StreamResult]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def stream(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        on_content: OnContent | None = None,
        on_thinking: OnThinking | None = None,
        idle_timeout_s: float = 120.0,
        hard_timeout_s: float = 600.0,
    ) -> StreamResult:
        self.calls.append(
            {"system": system, "messages": list(messages), "tools": [t.name for t in tools]}
        )
        return self._results.pop(0)


class _SlowProvider(Provider):
    def __init__(self, delay_s: float, result: StreamResult) -> None:
        self._delay_s = delay_s
        self._result = result

    async def stream(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        on_content: OnContent | None = None,
        on_thinking: OnThinking | None = None,
        idle_timeout_s: float = 120.0,
        hard_timeout_s: float = 600.0,
    ) -> StreamResult:
        await asyncio.sleep(self._delay_s)
        return self._result


async def test_run_sub_agent_returns_error_for_unknown_profile(
    tmp_path: Path, agent_manager: AgentManager
) -> None:
    provider = _FakeProvider([])
    result = await run_sub_agent(
        task_id="t1",
        task="do something",
        agent="does-not-exist",
        tier=ModelTier.FAST,
        agent_manager=agent_manager,
        router=_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=_make_registry_with_echo_and_spawn_agents(tmp_path),
        host_shell="zsh on darwin",
        deadline_s=600.0,
    )
    assert result.success is False
    assert result.error is not None
    assert "does-not-exist" in result.error


async def test_run_sub_agent_returns_final_text_on_success(
    tmp_path: Path, agent_manager: AgentManager
) -> None:
    provider = _FakeProvider([StreamResult(success=True, content="调研结论：一切正常")])
    result = await run_sub_agent(
        task_id="t1",
        task="调研一下",
        agent="researcher",
        tier=ModelTier.FAST,
        agent_manager=agent_manager,
        router=_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=_make_registry_with_echo_and_spawn_agents(tmp_path),
        host_shell="zsh on darwin",
        deadline_s=600.0,
    )
    assert result.success is True
    assert result.content == "调研结论：一切正常"
    assert result.error is None
    assert result.rounds == 0


async def test_run_sub_agent_defaults_to_researcher_when_agent_is_none(
    tmp_path: Path, agent_manager: AgentManager
) -> None:
    provider = _FakeProvider([StreamResult(success=True, content="ok")])
    result = await run_sub_agent(
        task_id="t1",
        task="task",
        agent=None,
        tier=ModelTier.FAST,
        agent_manager=agent_manager,
        router=_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=_make_registry_with_echo_and_spawn_agents(tmp_path),
        host_shell="zsh on darwin",
        deadline_s=600.0,
    )
    assert result.success is True


async def test_run_sub_agent_reports_provider_error(
    tmp_path: Path, agent_manager: AgentManager
) -> None:
    provider = _FakeProvider([StreamResult(success=False, error="request_idle_timeout")])
    result = await run_sub_agent(
        task_id="t1",
        task="task",
        agent="researcher",
        tier=ModelTier.FAST,
        agent_manager=agent_manager,
        router=_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=_make_registry_with_echo_and_spawn_agents(tmp_path),
        host_shell="zsh on darwin",
        deadline_s=600.0,
    )
    assert result.success is False
    assert result.error == "模型响应中断，可能是网络问题"


async def test_run_sub_agent_reports_budget_exhausted_as_failure(
    tmp_path: Path, agent_manager: AgentManager
) -> None:
    operator = agent_manager.resolve_profile("operator")
    assert operator is not None
    agent_manager.update_agent(operator.id, max_rounds=1)

    provider = _FakeProvider(
        [
            StreamResult(
                success=True, tool_uses=[ToolUseBlock(id="1", name="echo_tool", input={})]
            ),
            StreamResult(
                success=True, tool_uses=[ToolUseBlock(id="2", name="echo_tool", input={})]
            ),
        ]
    )
    result = await run_sub_agent(
        task_id="t1",
        task="task",
        agent="operator",
        tier=ModelTier.FAST,
        agent_manager=agent_manager,
        router=_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=_make_registry_with_echo_and_spawn_agents(tmp_path),
        host_shell="zsh on darwin",
        deadline_s=600.0,
    )
    assert result.success is False
    assert result.error is not None
    assert "回合预算耗尽" in result.error
    assert result.rounds == 1


async def test_run_sub_agent_reports_time_exhausted_with_nonempty_content(
    tmp_path: Path, agent_manager: AgentManager
) -> None:
    """子 agent 命中自己的软限时（deadline_s 扣除安全边界后 <= 0）时，应保留已生成的部分内容，
    而不是像真正卡死超时那样被外层 cancel() 丢弃——区别于上面的回合预算耗尽用例（那个是
    success=False），这里 success 应为 True，因为模型确实产出了一段可用的回答。
    """
    provider = _FakeProvider(
        [
            StreamResult(
                success=True,
                content="部分调研结果",
                tool_uses=[ToolUseBlock(id="1", name="echo_tool", input={})],
            ),
        ]
    )
    result = await run_sub_agent(
        task_id="t1",
        task="调研一下",
        agent="researcher",
        tier=ModelTier.FAST,
        agent_manager=agent_manager,
        router=_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=_make_registry_with_echo_and_spawn_agents(tmp_path),
        host_shell="zsh on darwin",
        deadline_s=1.0,
    )
    assert result.success is True
    assert result.content == "部分调研结果"
    assert result.error == "时间预算耗尽，已提前收尾"
    assert len(provider.calls) == 1


async def test_run_sub_agent_includes_host_shell_in_system_prompt(
    tmp_path: Path, agent_manager: AgentManager
) -> None:
    provider = _FakeProvider([StreamResult(success=True, content="done")])
    await run_sub_agent(
        task_id="t1",
        task="task",
        agent="researcher",
        tier=ModelTier.FAST,
        agent_manager=agent_manager,
        router=_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=_make_registry_with_echo_and_spawn_agents(tmp_path),
        host_shell="zsh on darwin",
        deadline_s=600.0,
    )
    assert "zsh on darwin" in provider.calls[0]["system"]


async def test_run_sub_agent_excludes_spawn_agents_tool_even_for_allow_all_profile(
    tmp_path: Path, agent_manager: AgentManager
) -> None:
    provider = _FakeProvider([StreamResult(success=True, content="done")])
    await run_sub_agent(
        task_id="t1",
        task="task",
        agent="operator",
        tier=ModelTier.FAST,
        agent_manager=agent_manager,
        router=_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=_make_registry_with_echo_and_spawn_agents(tmp_path),
        host_shell="zsh on darwin",
        deadline_s=600.0,
    )
    tool_names = provider.calls[0]["tools"]
    assert "spawn_agents" not in tool_names
    assert "echo_tool" in tool_names


async def test_register_spawn_agents_tool_runs_multiple_tasks_in_parallel(
    tmp_path: Path, agent_manager: AgentManager
) -> None:
    registry = _make_bare_registry(tmp_path)
    provider = _FakeProvider(
        [
            StreamResult(success=True, content="结果A"),
            StreamResult(success=True, content="结果B"),
        ]
    )
    register_spawn_agents_tool(
        agent_manager=agent_manager,
        router=_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=registry,
        host_shell="zsh on darwin",
    )

    result = await registry.execute(
        ToolUseBlock(
            id="call1",
            name="spawn_agents",
            input={
                "tasks": [
                    {"id": "a", "task": "task a", "agent": "researcher"},
                    {"id": "b", "task": "task b", "agent": "researcher"},
                ]
            },
        ),
        session_id="s1",
    )

    assert result.is_error is False
    payload = json.loads(result.content)
    assert payload["timed_out"] is False
    results_by_id = {r["id"]: r for r in payload["results"]}
    assert results_by_id["a"]["success"] is True
    assert results_by_id["a"]["content"] == "结果A"
    assert results_by_id["b"]["success"] is True
    assert results_by_id["b"]["content"] == "结果B"


async def test_register_spawn_agents_tool_rejects_too_few_tasks(
    tmp_path: Path, agent_manager: AgentManager
) -> None:
    registry = _make_bare_registry(tmp_path)
    register_spawn_agents_tool(
        agent_manager=agent_manager,
        router=_router(),
        providers={ProviderName.ANTHROPIC: _FakeProvider([])},
        registry=registry,
        host_shell="zsh on darwin",
    )

    result = await registry.execute(
        ToolUseBlock(
            id="call1", name="spawn_agents", input={"tasks": [{"id": "a", "task": "x"}]}
        ),
        session_id="s1",
    )

    assert result.is_error is True
    assert "数量必须" in result.content


async def test_register_spawn_agents_tool_rejects_invalid_model_tier(
    tmp_path: Path, agent_manager: AgentManager
) -> None:
    registry = _make_bare_registry(tmp_path)
    register_spawn_agents_tool(
        agent_manager=agent_manager,
        router=_router(),
        providers={ProviderName.ANTHROPIC: _FakeProvider([])},
        registry=registry,
        host_shell="zsh on darwin",
    )

    result = await registry.execute(
        ToolUseBlock(
            id="call1",
            name="spawn_agents",
            input={
                "tasks": [
                    {"id": "a", "task": "x", "model_tier": "not-a-real-tier"},
                    {"id": "b", "task": "y"},
                ]
            },
        ),
        session_id="s1",
    )

    assert result.is_error is True
    assert "not-a-real-tier" in result.content


async def test_register_spawn_agents_tool_rejects_missing_required_fields(
    tmp_path: Path, agent_manager: AgentManager
) -> None:
    registry = _make_bare_registry(tmp_path)
    register_spawn_agents_tool(
        agent_manager=agent_manager,
        router=_router(),
        providers={ProviderName.ANTHROPIC: _FakeProvider([])},
        registry=registry,
        host_shell="zsh on darwin",
    )

    result = await registry.execute(
        ToolUseBlock(
            id="call1", name="spawn_agents", input={"tasks": [{"id": "a"}, {"id": "b"}]}
        ),
        session_id="s1",
    )

    assert result.is_error is True


async def test_register_spawn_agents_tool_cancels_pending_tasks_on_timeout(
    tmp_path: Path, agent_manager: AgentManager
) -> None:
    registry = _make_bare_registry(tmp_path)
    fast_provider = _FakeProvider([StreamResult(success=True, content="快的结果")])
    slow_provider = _SlowProvider(1.0, StreamResult(success=True, content="慢的结果"))
    register_spawn_agents_tool(
        agent_manager=agent_manager,
        router=_multi_provider_router(),
        providers={ProviderName.ANTHROPIC: fast_provider, ProviderName.OPENAI: slow_provider},
        registry=registry,
        host_shell="zsh on darwin",
        deadline_s=0.05,
    )

    result = await registry.execute(
        ToolUseBlock(
            id="call1",
            name="spawn_agents",
            input={
                "tasks": [
                    {"id": "fast", "task": "x", "agent": "researcher"},
                    {
                        "id": "slow",
                        "task": "y",
                        "agent": "researcher",
                        "model_tier": "medium",
                    },
                ]
            },
        ),
        session_id="s1",
    )

    assert result.is_error is False
    payload = json.loads(result.content)
    assert payload["timed_out"] is True
    results_by_id = {r["id"]: r for r in payload["results"]}
    assert results_by_id["fast"]["success"] is True
    assert results_by_id["fast"]["content"] == "快的结果"
    assert results_by_id["slow"]["success"] is False
    assert results_by_id["slow"]["error"] == "超时，已取消"
