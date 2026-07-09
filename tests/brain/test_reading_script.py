"""``generate_reading_script`` 的正常路径、空内容报错与 ``NoModelAvailableError`` 透传回归测试。

用假 ``Provider``（同 ``test_loop.py`` 的 ``_FakeProvider`` 手法）直接返回固定 ``StreamResult``，
不发真实网络请求。
"""

from __future__ import annotations

from typing import Any

import pytest

from miku_on_desk.brain.model_router import ModelRouter, NoModelAvailableError
from miku_on_desk.brain.providers.base import (
    Message,
    OnContent,
    OnThinking,
    Provider,
    StreamResult,
    ToolDefinition,
)
from miku_on_desk.brain.reading_script import ReadingScriptError, generate_reading_script
from miku_on_desk.config.settings import ModelRouterConfig, ModelTier, ProviderConfig, ProviderName


class _FakeProvider(Provider):
    def __init__(self, result: StreamResult) -> None:
        self._result = result
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
        self.calls.append({"model": model, "system": system, "messages": messages, "tools": tools})
        return self._result


def _router_with_anthropic(model_id: str = "claude-mini") -> ModelRouter:
    config = ModelRouterConfig()
    config.anthropic = ProviderConfig(api_key="key", models={ModelTier.MINI: model_id})
    return ModelRouter(config)


async def test_generate_reading_script_returns_stripped_content() -> None:
    provider = _FakeProvider(StreamResult(success=True, content="  你好呀，今天天气真不错。  "))
    router = _router_with_anthropic()

    text = await generate_reading_script(
        description="一个爱笑的猫娘", router=router, providers={ProviderName.ANTHROPIC: provider}
    )

    assert text == "你好呀，今天天气真不错。"
    assert len(provider.calls) == 1
    assert provider.calls[0]["model"] == "claude-mini"


async def test_generate_reading_script_passes_description_into_prompt() -> None:
    provider = _FakeProvider(StreamResult(success=True, content="占位文本"))
    router = _router_with_anthropic()

    await generate_reading_script(
        description="一个爱笑的猫娘", router=router, providers={ProviderName.ANTHROPIC: provider}
    )

    messages = provider.calls[0]["messages"]
    assert len(messages) == 1
    assert "一个爱笑的猫娘" in str(messages[0].content)


async def test_generate_reading_script_raises_on_empty_content() -> None:
    provider = _FakeProvider(StreamResult(success=False, content="   ", raw_error="配额耗尽"))
    router = _router_with_anthropic()

    with pytest.raises(ReadingScriptError, match="配额耗尽"):
        await generate_reading_script(
            description="任意描述", router=router, providers={ProviderName.ANTHROPIC: provider}
        )


async def test_generate_reading_script_propagates_no_model_available_error() -> None:
    router = ModelRouter(ModelRouterConfig())

    with pytest.raises(NoModelAvailableError):
        await generate_reading_script(description="任意描述", router=router, providers={})
