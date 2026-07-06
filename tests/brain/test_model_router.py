"""ModelRouter 的分层升级逃生逻辑回归测试。"""

from __future__ import annotations

import pytest

from miku_on_desk.brain.model_router import ModelRouter, NoModelAvailableError, ResolvedModel
from miku_on_desk.config.settings import ModelRouterConfig, ModelTier, ProviderConfig, ProviderName


def _config(**provider_models: dict[ModelTier, str]) -> ModelRouterConfig:
    config = ModelRouterConfig()
    for provider_name, models in provider_models.items():
        setattr(config, provider_name, ProviderConfig(api_key="k", models=models))
    return config


def test_resolve_returns_exact_tier_match() -> None:
    router = ModelRouter(_config(anthropic={ModelTier.FAST: "claude-haiku"}))
    assert router.resolve(ModelTier.FAST) == ResolvedModel(
        provider=ProviderName.ANTHROPIC, model_id="claude-haiku", tier=ModelTier.FAST
    )


def test_resolve_escalates_upward_when_requested_tier_missing() -> None:
    router = ModelRouter(_config(anthropic={ModelTier.HEAVY: "claude-opus"}))
    resolved = router.resolve(ModelTier.MINI)
    assert resolved.tier == ModelTier.HEAVY
    assert resolved.model_id == "claude-opus"


def test_resolve_never_downgrades() -> None:
    router = ModelRouter(_config(anthropic={ModelTier.MINI: "claude-haiku"}))
    with pytest.raises(NoModelAvailableError):
        router.resolve(ModelTier.HEAVY)


def test_resolve_raises_when_nothing_configured() -> None:
    router = ModelRouter(ModelRouterConfig())
    with pytest.raises(NoModelAvailableError):
        router.resolve(ModelTier.MEDIUM)


def test_resolve_ignores_disabled_providers_without_api_key() -> None:
    config = ModelRouterConfig()
    config.openai = ProviderConfig(api_key=None, models={ModelTier.FAST: "gpt-5-mini"})
    router = ModelRouter(config)
    with pytest.raises(NoModelAvailableError):
        router.resolve(ModelTier.FAST)


def test_resolve_picks_first_enabled_provider_when_multiple_configure_same_tier() -> None:
    config = _config(
        anthropic={ModelTier.FAST: "claude-haiku"},
        openai={ModelTier.FAST: "gpt-5-mini"},
    )
    resolved = ModelRouter(config).resolve(ModelTier.FAST)
    assert resolved.provider == ProviderName.ANTHROPIC


def test_resolve_provider_finds_model_within_single_provider_ignoring_others() -> None:
    config = _config(
        anthropic={ModelTier.FAST: "claude-haiku"},
        qwen={ModelTier.FAST: "qwen3-vl-plus"},
    )
    resolved = ModelRouter(config).resolve_provider(ProviderName.QWEN, ModelTier.FAST)
    assert resolved == ResolvedModel(
        provider=ProviderName.QWEN, model_id="qwen3-vl-plus", tier=ModelTier.FAST
    )


def test_resolve_provider_escalates_tiers_within_same_provider() -> None:
    config = _config(qwen={ModelTier.HEAVY: "qwen3-vl-max"})
    resolved = ModelRouter(config).resolve_provider(ProviderName.QWEN, ModelTier.MINI)
    assert resolved.tier == ModelTier.HEAVY
    assert resolved.model_id == "qwen3-vl-max"


def test_resolve_provider_raises_when_provider_disabled_or_unconfigured() -> None:
    router = ModelRouter(ModelRouterConfig())
    with pytest.raises(NoModelAvailableError):
        router.resolve_provider(ProviderName.QWEN, ModelTier.FAST)


def test_resolve_provider_raises_when_model_missing_at_or_above_tier() -> None:
    config = _config(qwen={ModelTier.MINI: "qwen3-vl-mini"})
    router = ModelRouter(config)
    with pytest.raises(NoModelAvailableError):
        router.resolve_provider(ProviderName.QWEN, ModelTier.HEAVY)


def test_resolve_fallback_returns_none_when_disabled() -> None:
    config = _config(
        anthropic={ModelTier.FAST: "claude-haiku"}, openai={ModelTier.FAST: "gpt-5-mini"}
    )
    router = ModelRouter(config)
    assert router.resolve_fallback(ModelTier.FAST, exclude=ProviderName.ANTHROPIC) is None


def test_resolve_fallback_finds_different_enabled_provider_at_same_tier() -> None:
    config = _config(
        anthropic={ModelTier.FAST: "claude-haiku"}, openai={ModelTier.FAST: "gpt-5-mini"}
    )
    config.enable_cross_provider_fallback = True
    router = ModelRouter(config)
    fallback = router.resolve_fallback(ModelTier.FAST, exclude=ProviderName.ANTHROPIC)
    assert fallback == ResolvedModel(
        provider=ProviderName.OPENAI, model_id="gpt-5-mini", tier=ModelTier.FAST
    )


def test_resolve_fallback_escalates_tiers_when_same_tier_unavailable_elsewhere() -> None:
    config = _config(
        anthropic={ModelTier.FAST: "claude-haiku"}, openai={ModelTier.HEAVY: "gpt-5"}
    )
    config.enable_cross_provider_fallback = True
    router = ModelRouter(config)
    fallback = router.resolve_fallback(ModelTier.FAST, exclude=ProviderName.ANTHROPIC)
    assert fallback == ResolvedModel(
        provider=ProviderName.OPENAI, model_id="gpt-5", tier=ModelTier.HEAVY
    )


def test_resolve_fallback_returns_none_when_only_excluded_provider_configured() -> None:
    config = _config(anthropic={ModelTier.FAST: "claude-haiku"})
    config.enable_cross_provider_fallback = True
    router = ModelRouter(config)
    assert router.resolve_fallback(ModelTier.FAST, exclude=ProviderName.ANTHROPIC) is None
