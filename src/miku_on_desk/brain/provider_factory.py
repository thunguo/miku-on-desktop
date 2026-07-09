"""按配置组装 LLM Provider 实例。

从 `main.py` 抽出（brain 后台线程与 UI 侧的朗读文本生成都需要一套 Provider，
但 `face/*` 不能反向 import `main.py`，且不同线程不应共享同一份 Provider 实例）。
"""

from __future__ import annotations

from miku_on_desk.brain.providers.anthropic_provider import AnthropicProvider
from miku_on_desk.brain.providers.base import Provider
from miku_on_desk.brain.providers.gemini_provider import GeminiProvider
from miku_on_desk.brain.providers.openai_compatible_provider import OpenAICompatibleProvider
from miku_on_desk.config.settings import ModelRouterConfig, ProviderName


def build_providers(config: ModelRouterConfig) -> dict[ProviderName, Provider]:
    providers: dict[ProviderName, Provider] = {}
    for name in config.enabled_providers():
        provider_config = config.provider(name)
        api_key = provider_config.api_key
        assert api_key is not None
        if name is ProviderName.ANTHROPIC:
            providers[name] = AnthropicProvider(api_key=api_key, base_url=provider_config.base_url)
        elif name is ProviderName.OPENAI or name is ProviderName.QWEN:
            providers[name] = OpenAICompatibleProvider(
                api_key=api_key, base_url=provider_config.base_url
            )
        else:
            providers[name] = GeminiProvider(api_key=api_key, base_url=provider_config.base_url)
    return providers
