"""ElevenLabs TTS 实现：走官方 ``elevenlabs`` SDK 的 ``text_to_speech.convert`` 合成音频。

字段映射：``config.voice`` → voice_id，``config.model`` → model_id，``config.api_key`` →
鉴权，``config.base_url`` 可覆盖默认站点（自建/代理时用）。强制请求 mp3，与其它引擎对齐，
保证下游 ``QMediaPlayer`` 直接可播。SDK 的异步 ``convert`` 返回音频字节的 ``AsyncIterator``，
这里拼接成完整字节返回。
"""

from __future__ import annotations

from elevenlabs.client import AsyncElevenLabs

from miku_on_desk.config.settings import TTSConfig

_DEFAULT_MODEL = "eleven_multilingual_v2"
_OUTPUT_FORMAT = "mp3_44100_128"


class ElevenLabsTTSProvider:
    """把文本合成为 mp3 字节的 :class:`~miku_on_desk.brain.tts.base.TTSProvider` 实现。"""

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

    async def synthesize(self, text: str) -> bytes:
        stream = self._client.text_to_speech.convert(
            self._voice_id,
            text=text,
            model_id=self._model,
            output_format=_OUTPUT_FORMAT,
        )
        buffer = bytearray()
        async for chunk in stream:
            buffer.extend(chunk)
        return bytes(buffer)
