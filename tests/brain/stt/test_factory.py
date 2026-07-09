"""STT 工厂按 config.provider 分发到正确实现的回归测试。"""

from __future__ import annotations

import pytest

from miku_on_desk.brain.stt.elevenlabs_provider import ElevenLabsSTTProvider
from miku_on_desk.brain.stt.factory import create_stt_provider
from miku_on_desk.config.settings import VoiceInputConfig, VoiceInputProviderName


def test_create_stt_provider_builds_elevenlabs_by_default() -> None:
    config = VoiceInputConfig(api_key="sk-el")

    provider = create_stt_provider(config)

    assert isinstance(provider, ElevenLabsSTTProvider)


def test_create_stt_provider_elevenlabs_without_api_key_raises() -> None:
    config = VoiceInputConfig(provider=VoiceInputProviderName.ELEVENLABS)

    with pytest.raises(ValueError, match="api_key"):
        create_stt_provider(config)
