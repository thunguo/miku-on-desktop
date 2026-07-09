"""edge-tts 实现：调用微软 Edge 的在线神经语音服务合成音频。

免 API Key、中文自然度好，但**必须联网**、且是非官方接口。逐块产出 mp3 字节——
该协议只提供 mp3 系输出，没有裸 PCM 选项（``edge_tts.constants`` 里的可选格式列表
均为 mp3/webm 容器），因此 ``pcm_format`` 始终为 ``None``，播放侧走整句缓冲兜底路径。
语音音库名与 ``rate``/``volume`` 的相对量格式直接来自 edge-tts 约定，由 ``TTSConfig``
透传。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import edge_tts

from miku_on_desk.brain.tts.base import PcmFormat
from miku_on_desk.config.settings import TTSConfig


class EdgeTTSProvider:
    """把文本合成为 mp3 字节流的 :class:`~miku_on_desk.brain.tts.base.TTSProvider` 实现。"""

    def __init__(self, config: TTSConfig) -> None:
        self._voice = config.voice
        self._rate = config.rate
        self._volume = config.volume
        self.pcm_format: PcmFormat | None = None

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        communicate = edge_tts.Communicate(
            text, self._voice, rate=self._rate, volume=self._volume
        )
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
