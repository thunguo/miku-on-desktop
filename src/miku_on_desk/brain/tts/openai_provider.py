"""OpenAI 兼容 TTS 实现：调用 ``/v1/audio/speech`` 把文本合成为音频字节流。

只要接入点遵循 OpenAI 的 speech 协议（``model``/``voice``/``input`` 入参、二进制音频出参），
换个 ``base_url``/``api_key``/``model`` 就能接上不同厂商，无需改代码。强制
``response_format=mp3`` 与 edge 实现对齐。OpenAI 官方接口的 ``response_format`` 也支持
``"pcm"``，但"OpenAI 兼容"可能指向未知第三方后端，是否遵循同一 PCM 约定不可控，故本实现
先只做流式转发、暂不切 PCM；``pcm_format`` 始终为 ``None``，播放侧走整句缓冲兜底路径。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import openai

from miku_on_desk.brain.tts.base import PcmFormat
from miku_on_desk.config.settings import TTSConfig


class OpenAITTSProvider:
    """把文本合成为 mp3 字节流的 :class:`~miku_on_desk.brain.tts.base.TTSProvider` 实现。"""

    def __init__(self, config: TTSConfig) -> None:
        if not config.api_key:
            raise ValueError("OpenAI 兼容 TTS 需要配置 api_key")
        self._client = openai.AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self._model = config.model
        self._voice = config.voice
        self.pcm_format: PcmFormat | None = None

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        async with self._client.audio.speech.with_streaming_response.create(
            model=self._model,
            voice=self._voice,
            input=text,
            response_format="mp3",
        ) as response:
            async for chunk in response.iter_bytes():
                if chunk:
                    yield chunk
