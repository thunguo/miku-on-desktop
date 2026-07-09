"""``run_ai_loop`` 测试用假 Provider 与最小 ToolRegistry 构造，从 ``tests/brain/test_loop.py``
提炼而来，供该文件与 ``tests/eval/**`` 共同复用。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miku_on_desk.brain.model_router import ModelRouter
from miku_on_desk.brain.providers.base import (
    Message,
    OnContent,
    OnThinking,
    Provider,
    StreamResult,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from miku_on_desk.brain.tools.path_sandbox import PathSandbox
from miku_on_desk.brain.tools.policy import Decision, PolicyEngine, ToolPolicySpec
from miku_on_desk.brain.tools.read_tracker import ReadTracker
from miku_on_desk.brain.tools.registry import ToolRegistration, ToolRegistry
from miku_on_desk.config.settings import ModelRouterConfig, ModelTier, ProviderConfig

SESSION = "session-1"
TIER = ModelTier.FAST


class FakeProvider(Provider):
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
            {
                "model": model,
                "system": system,
                "messages": [message.model_copy(deep=True) for message in messages],
                "idle_timeout_s": idle_timeout_s,
                "hard_timeout_s": hard_timeout_s,
            }
        )
        if on_content is not None:
            on_content("streamed-content")
        if on_thinking is not None:
            on_thinking("streamed-thinking")
        return self._results.pop(0)


async def never_confirm(_tool_use: ToolUseBlock, _reason: str | None) -> bool:
    raise AssertionError("confirm 在这个测试场景下不应该被调用")


async def _ok_handler(_input: dict[str, Any]) -> str:
    return "ok"


def tool_use(tool_use_id: str, name: str = "echo_tool") -> ToolUseBlock:
    return ToolUseBlock(id=tool_use_id, name=name, input={})


def make_registry(tmp_path: Path, *, trusted_mode: bool = True) -> ToolRegistry:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
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
    registry.register(
        ToolRegistration(
            definition=ToolDefinition(name="echo_tool", description="d", input_schema={}),
            handler=_ok_handler,
        )
    )
    return registry


def make_confirmation_registry(tmp_path: Path) -> ToolRegistry:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    sandbox = PathSandbox(cwd=cwd, output_dir=tmp_path / "output", data_dir=tmp_path / "data")
    policy = PolicyEngine(
        trusted_mode=False,
        allowed_tools=frozenset(),
        denied_tools=frozenset(),
        default_decision=Decision.ALLOW,
        path_sandbox=sandbox,
        read_tracker=ReadTracker(),
    )
    registry = ToolRegistry(policy, ReadTracker())
    registry.register(
        ToolRegistration(
            definition=ToolDefinition(name="dangerous_tool", description="d", input_schema={}),
            handler=_ok_handler,
            policy_spec=ToolPolicySpec(requires_confirmation=True),
        )
    )
    return registry


def build_router(model_id: str = "model-x") -> ModelRouter:
    config = ModelRouterConfig()
    config.anthropic = ProviderConfig(api_key="key", models={TIER: model_id})
    return ModelRouter(config)


def build_router_with_fallback(
    *,
    model_id: str = "model-x",
    fallback_model_id: str = "model-y",
    enabled: bool = True,
) -> ModelRouter:
    """两个 Provider（anthropic 为主、openai 为备）都配置了同一层级的模型；
    ``enabled`` 控制 ``enable_cross_provider_fallback`` 开关，默认打开以便测试降级路径，
    传 ``enabled=False`` 可以复用同一个 helper 测试"开关关闭时不降级"。
    """
    config = ModelRouterConfig()
    config.anthropic = ProviderConfig(api_key="key", models={TIER: model_id})
    config.openai = ProviderConfig(api_key="key2", models={TIER: fallback_model_id})
    config.enable_cross_provider_fallback = enabled
    return ModelRouter(config)


def tool_results_by_id(messages: list[Message]) -> dict[str, ToolResultBlock]:
    return {
        block.tool_use_id: block
        for message in messages
        if isinstance(message.content, list)
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }
