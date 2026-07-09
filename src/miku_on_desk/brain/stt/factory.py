"""按配置挑选并构建 STT 引擎。惰性 import 各 provider 实现——避免在未启用语音输入时也要
加载 ElevenLabs SDK 及其依赖。
"""

from __future__ import annotations

from collections.abc import Callable

from miku_on_desk.brain.stt.base import STTProvider
from miku_on_desk.config.settings import VoiceInputConfig, VoiceInputProviderName


def _build_elevenlabs(config: VoiceInputConfig) -> STTProvider:
    from miku_on_desk.brain.stt.elevenlabs_provider import ElevenLabsSTTProvider

    return ElevenLabsSTTProvider(config)


_BUILDERS: dict[VoiceInputProviderName, Callable[[VoiceInputConfig], STTProvider]] = {
    VoiceInputProviderName.ELEVENLABS: _build_elevenlabs,
}


def create_stt_provider(config: VoiceInputConfig) -> STTProvider:
    return _BUILDERS[config.provider](config)
