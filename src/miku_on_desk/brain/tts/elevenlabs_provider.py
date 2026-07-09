"""ElevenLabs TTS 实现：走官方 ``elevenlabs`` SDK 的 ``text_to_speech.convert`` 合成音频。

字段映射：``config.voice`` → voice_id，``config.model`` → model_id，``config.api_key`` →
鉴权，``config.base_url`` 可覆盖默认站点（自建/代理时用）。请求裸 PCM（16-bit 有符号
小端、单声道、24kHz——``pcm_24000`` 不受订阅档位限制），播放侧据此可直接推流到
``QAudioSink`` 而无需解码，实现句内真流式播放。SDK 的异步 ``convert`` 本身就是按 HTTP
chunk 逐块产出的 ``AsyncIterator``，这里原样转发，不做任何攒批缓冲。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from elevenlabs.client import AsyncElevenLabs

from miku_on_desk.brain.tts.base import PcmFormat
from miku_on_desk.config.settings import TTSConfig

_DEFAULT_MODEL = "eleven_multilingual_v2"
_OUTPUT_FORMAT = "pcm_24000"
_SAMPLE_RATE = 24000


class ElevenLabsTTSProvider:
    """把文本合成为 PCM 字节流的 :class:`~miku_on_desk.brain.tts.base.TTSProvider` 实现。"""

    def __init__(self, config: TTSConfig) -> None:
        if not config.api_key:
            raise ValueError("ElevenLabs TTS 需要配置 api_key")
        if not config.voice:
            raise ValueError("ElevenLabs TTS 需要配置 voice（voice_id）")
        self._voice_id = config.voice
        # 共享的 model 字段默认是 OpenAI 的 "tts-1"，对 ElevenLabs 无意义，遇到它就回退到
        # ElevenLabs 的多语种默认模型，省得用户切引擎后还要手动改 model。
        self._model = config.model if config.model != "tts-1" else _DEFAULT_MODEL
        # base_url 留空时 SDK 用官方默认站点；传 None 即为不覆盖。
        self._client = AsyncElevenLabs(
            api_key=config.api_key, base_url=config.base_url or None
        )
        self.pcm_format: PcmFormat | None = PcmFormat(sample_rate=_SAMPLE_RATE)

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        stream = self._client.text_to_speech.convert(
            self._voice_id,
            text=text,
            model_id=self._model,
            output_format=_OUTPUT_FORMAT,
        )
        async for chunk in stream:
            if chunk:
                yield chunk
