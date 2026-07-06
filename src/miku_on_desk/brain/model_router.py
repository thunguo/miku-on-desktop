"""模型分层路由：从用户实际配置、启用的 Provider 里组装 tier -> (provider, model_id) 路由表。

"只升级不降级"：请求 mini 层但用户没配置 mini 模型时，按 mini→fast→medium→heavy 顺序向上找
第一个有配置的层级，而不是报错或静默换成一个更弱的模型——弱模型在复杂任务上的失败通常比
慢/贵更难排查。不做"偏好当前 provider 以避免新建 SDK client"这类微优化：每个 Provider 实例
本身就是长期持有的对象，切换 provider 不需要重新建连接，这里没有对应的成本可省。
"""

from __future__ import annotations

from dataclasses import dataclass

from miku_on_desk.config.settings import ModelRouterConfig, ModelTier, ProviderName

_TIER_ORDER = [ModelTier.MINI, ModelTier.FAST, ModelTier.MEDIUM, ModelTier.HEAVY]


@dataclass(frozen=True)
class ResolvedModel:
    provider: ProviderName
    model_id: str
    tier: ModelTier
    """实际命中的层级，可能高于请求的层级（升级逃生的结果）。"""


class NoModelAvailableError(RuntimeError):
    """所有层级都没有任何已启用 Provider 配置模型。

    调用方应停止本轮 AI 循环，提示用户去设置面板配置 Provider 凭证与模型。
    """


class ModelRouter:
    def __init__(self, config: ModelRouterConfig) -> None:
        self._config = config

    def resolve(self, tier: ModelTier) -> ResolvedModel:
        start_index = _TIER_ORDER.index(tier)
        for candidate_tier in _TIER_ORDER[start_index:]:
            for provider_name in self._config.enabled_providers():
                model_id = self._config.provider(provider_name).models.get(candidate_tier)
                if model_id:
                    return ResolvedModel(
                        provider=provider_name, model_id=model_id, tier=candidate_tier
                    )
        raise NoModelAvailableError(
            f"没有任何已启用的 Provider 为 {tier.value} 或更高层级配置模型"
        )

    def resolve_provider(self, provider_name: ProviderName, tier: ModelTier) -> ResolvedModel:
        """与 `resolve()` 同样的升级逃生查找，但只在指定的单个 Provider 内进行。

        用于"这里必须是某个特定 Provider，不能被其它同样在该层级配置了模型的 Provider
        抢先命中"的场景（如 screen_analyze 的视觉定位必须是 Qwen）——不遍历
        `enabled_providers()`，所以结果完全不受 Provider 枚举声明顺序影响。
        """
        if not self._config.provider(provider_name).enabled:
            raise NoModelAvailableError(f'Provider "{provider_name.value}" 未启用')
        start_index = _TIER_ORDER.index(tier)
        for candidate_tier in _TIER_ORDER[start_index:]:
            model_id = self._config.provider(provider_name).models.get(candidate_tier)
            if model_id:
                return ResolvedModel(provider=provider_name, model_id=model_id, tier=candidate_tier)
        raise NoModelAvailableError(
            f'Provider "{provider_name.value}" 未在 {tier.value} 或更高层级配置模型'
        )

    def resolve_fallback(self, tier: ModelTier, *, exclude: ProviderName) -> ResolvedModel | None:
        """跨 Provider 降级：`exclude` 指定的 Provider 已经重试耗尽后，找一个不同的、在该
        层级或更高层级配置了模型的已启用 Provider。

        与 `resolve()` 不同，找不到时返回 ``None`` 而不是抛异常——原始 Provider 的失败本身
        已经是可报告的终态错误，降级只是"锦上添花"，找不到就该让原始错误照常向上传播，而不是
        再制造一个新的异常掩盖它。`enable_cross_provider_fallback` 关闭时直接返回 ``None``。
        """
        if not self._config.enable_cross_provider_fallback:
            return None
        start_index = _TIER_ORDER.index(tier)
        for candidate_tier in _TIER_ORDER[start_index:]:
            for provider_name in self._config.enabled_providers():
                if provider_name == exclude:
                    continue
                model_id = self._config.provider(provider_name).models.get(candidate_tier)
                if model_id:
                    return ResolvedModel(
                        provider=provider_name, model_id=model_id, tier=candidate_tier
                    )
        return None
