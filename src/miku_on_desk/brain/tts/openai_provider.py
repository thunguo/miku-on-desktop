"""OpenAI 兼容 TTS 实现：调用 ``/v1/audio/speech`` 把文本合成为音频字节。

只要接入点遵循 OpenAI 的 speech 协议（``model``/``voice``/``input`` 入参、二进制音频出参），
换个 ``base_url``/``api_key``/``model`` 就能接上不同厂商，无需改代码。强制 ``response_format=mp3``
与 edge 实现对齐，保证下游 ``QMediaPlayer`` 能直接播放。
"""

from __future__ import annotations

import openai

from miku_on_desk.config.settings import TTSConfig


class OpenAITTSProvider:
    """把文本合成为 mp3 字节的 :class:`~miku_on_desk.brain.tts.base.TTSProvider` 实现。"""

    def __init__(self, config: TTSConfig) -> None:
        if not config.api_key:
            raise ValueError("OpenAI 兼容 TTS 需要配置 api_key")
        self._client = openai.AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self._model = config.model
        self._voice = config.voice

    async def synthesize(self, text: str) -> bytes:
        async with self._client.audio.speech.with_streaming_response.create(
            model=self._model,
            voice=self._voice,
            input=text,
            response_format="mp3",
        ) as response:
            buffer = bytearray()
            async for chunk in response.iter_bytes():
                buffer.extend(chunk)
            return bytes(buffer)
