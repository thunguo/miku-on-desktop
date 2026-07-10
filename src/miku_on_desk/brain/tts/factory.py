"""按配置挑选并构建 TTS 引擎。

新增一个引擎的完整改动面：``config.settings.TTSProviderName`` 加一项 + 写一个实现
``TTSProvider`` 协议的类 + 在这里的 ``_BUILDERS`` 注册一行。调用方（main.py）永远只跟
:func:`create_tts_provider` 打交道，不感知具体实现。实现按需惰性导入，避免为没选中的引擎
拉起 edge-tts / openai 依赖。
"""

from __future__ import annotations

from collections.abc import Callable

from miku_on_desk.brain.tts.base import TTSProvider
from miku_on_desk.brain.tts.fallback_provider import FallbackTTSProvider
from miku_on_desk.config.settings import TTSConfig, TTSProviderName


def _build_edge(config: TTSConfig) -> TTSProvider:
    from miku_on_desk.brain.tts.edge_provider import EdgeTTSProvider

    return EdgeTTSProvider(config)


def _build_openai(config: TTSConfig) -> TTSProvider:
    from miku_on_desk.brain.tts.openai_provider import OpenAITTSProvider

    return OpenAITTSProvider(config)


def _build_elevenlabs(config: TTSConfig) -> TTSProvider:
    from miku_on_desk.brain.tts.elevenlabs_provider import ElevenLabsTTSProvider

    return ElevenLabsTTSProvider(config)


_BUILDERS: dict[TTSProviderName, Callable[[TTSConfig], TTSProvider]] = {
    TTSProviderName.EDGE: _build_edge,
    TTSProviderName.OPENAI: _build_openai,
    TTSProviderName.ELEVENLABS: _build_elevenlabs,
}


def create_tts_provider(config: TTSConfig) -> TTSProvider:
    """根据 ``config.provider`` 构建对应 TTS 引擎。

    未知引擎会 ``KeyError``——枚举与 ``_BUILDERS`` 只要同步维护就不会发生；各引擎自身的
    配置校验（如 OpenAI 需要 api_key）由实现的构造函数负责，可能抛 ``ValueError``。

    ``config.fallback_to_edge`` 开启且当前引擎不是 Edge 本身时，用
    :class:`~miku_on_desk.brain.tts.fallback_provider.FallbackTTSProvider` 包装一层，
    合成失败时自动换 Edge 说完——见 :class:`TTSConfig` 文档说明的延迟权衡。
    """
    provider = _BUILDERS[config.provider](config)
    if config.fallback_to_edge and config.provider is not TTSProviderName.EDGE:
        return FallbackTTSProvider([provider, _build_edge(config)])
    return provider
