"""edge-tts 实现：调用微软 Edge 的在线神经语音服务合成音频。

免 API Key、中文自然度好，但**必须联网**、且是非官方接口。返回 mp3 字节。语音音库名与
``rate``/``volume`` 的相对量格式直接来自 edge-tts 约定，由 ``TTSConfig`` 透传。
"""

from __future__ import annotations

import edge_tts

from miku_on_desk.config.settings import TTSConfig


class EdgeTTSProvider:
    """把文本合成为 mp3 字节的 :class:`~miku_on_desk.brain.tts.base.TTSProvider` 实现。"""

    def __init__(self, config: TTSConfig) -> None:
        self._voice = config.voice
        self._rate = config.rate
        self._volume = config.volume

    async def synthesize(self, text: str) -> bytes:
        communicate = edge_tts.Communicate(
            text, self._voice, rate=self._rate, volume=self._volume
        )
        buffer = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.extend(chunk["data"])
        return bytes(buffer)
