"""TTS 工厂按 config.provider 分发到正确实现的回归测试。"""

from __future__ import annotations

import pytest

from miku_on_desk.brain.tts.edge_provider import EdgeTTSProvider
from miku_on_desk.brain.tts.elevenlabs_provider import ElevenLabsTTSProvider
from miku_on_desk.brain.tts.factory import create_tts_provider
from miku_on_desk.brain.tts.openai_provider import OpenAITTSProvider
from miku_on_desk.config.settings import TTSConfig, TTSProviderName


def test_create_tts_provider_builds_edge_by_default() -> None:
    provider = create_tts_provider(TTSConfig())

    assert isinstance(provider, EdgeTTSProvider)


def test_create_tts_provider_builds_openai_when_selected() -> None:
    config = TTSConfig(provider=TTSProviderName.OPENAI, api_key="sk-tts")

    provider = create_tts_provider(config)

    assert isinstance(provider, OpenAITTSProvider)


def test_create_tts_provider_openai_without_api_key_raises() -> None:
    config = TTSConfig(provider=TTSProviderName.OPENAI)

    with pytest.raises(ValueError, match="api_key"):
        create_tts_provider(config)


def test_create_tts_provider_builds_elevenlabs_when_selected() -> None:
    config = TTSConfig(
        provider=TTSProviderName.ELEVENLABS, api_key="sk-el", voice="21m00Tcm4TlvDq8ikWAM"
    )

    provider = create_tts_provider(config)

    assert isinstance(provider, ElevenLabsTTSProvider)


def test_create_tts_provider_elevenlabs_without_api_key_raises() -> None:
    config = TTSConfig(provider=TTSProviderName.ELEVENLABS)

    with pytest.raises(ValueError, match="api_key"):
        create_tts_provider(config)


def test_elevenlabs_provider_falls_back_from_openai_default_model() -> None:
    # 共享 model 字段默认 "tts-1"（OpenAI 的），ElevenLabs 应回退到多语种默认模型
    provider = ElevenLabsTTSProvider(
        TTSConfig(provider=TTSProviderName.ELEVENLABS, api_key="sk-el", voice="v1")
    )

    assert provider._model == "eleven_multilingual_v2"
