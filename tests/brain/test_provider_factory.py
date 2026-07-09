"""build_providers 按启用状态选择/构造对应 Provider 实现的回归测试。"""

from __future__ import annotations

from miku_on_desk.brain.provider_factory import build_providers
from miku_on_desk.brain.providers.anthropic_provider import AnthropicProvider
from miku_on_desk.brain.providers.gemini_provider import GeminiProvider
from miku_on_desk.brain.providers.openai_compatible_provider import OpenAICompatibleProvider
from miku_on_desk.config.settings import ModelRouterConfig, ModelTier, ProviderConfig, ProviderName


def test_build_providers_only_constructs_enabled_providers() -> None:
    config = ModelRouterConfig(
        anthropic=ProviderConfig(api_key="sk-ant", models={ModelTier.FAST: "haiku"}),
        openai=ProviderConfig(api_key=None, models={}),
        gemini=ProviderConfig(api_key="sk-gemini", models={ModelTier.HEAVY: "gemini-pro"}),
    )

    providers = build_providers(config)

    assert set(providers) == {ProviderName.ANTHROPIC, ProviderName.GEMINI}
    assert isinstance(providers[ProviderName.ANTHROPIC], AnthropicProvider)
    assert isinstance(providers[ProviderName.GEMINI], GeminiProvider)


def test_build_providers_constructs_openai_compatible_provider() -> None:
    config = ModelRouterConfig(
        openai=ProviderConfig(api_key="sk-openai", models={ModelTier.MEDIUM: "gpt-5"})
    )

    providers = build_providers(config)

    assert isinstance(providers[ProviderName.OPENAI], OpenAICompatibleProvider)
