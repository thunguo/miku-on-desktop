"""TTS 合成失败时按顺序尝试下一个候选 provider 的包装器。

统一整句缓冲，不做分片直通：``SpeechController`` 只在切换 provider 时读一次
``pcm_format`` 决定播放路径（裸 PCM 走 ``QAudioSink`` 连续流，压缩容器走整句缓冲 +
``QMediaPlayer``），同一实例的 chunk 不能中途改变格式解读方式。因此本类固定声明
``pcm_format = None``，对每个候选都先完整消费 ``synthesize_stream()`` 缓冲成一个
``bytes``，若该候选原生是裸 PCM 就用标准库 ``wave`` 包一层合法容器头再交出去——保证
不管最终哪个候选合成成功，吐出的都是 ``QMediaPlayer`` 能直接解码的自描述字节块。
代价是失去了实时逐块流式播放（比如 ElevenLabs），因此调用方应仅在用户显式开启降级时
才使用本包装器，而不是无条件包装所有 provider。
"""

from __future__ import annotations

import io
import logging
import wave
from collections.abc import AsyncIterator

from miku_on_desk.brain.tts.base import PcmFormat, TTSProvider

logger = logging.getLogger(__name__)


def _wrap_as_wav(data: bytes, fmt: PcmFormat) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(fmt.channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(fmt.sample_rate)
        wav_file.writeframes(data)
    return buffer.getvalue()


async def _synthesize_whole(provider: TTSProvider, text: str) -> bytes:
    chunks = [chunk async for chunk in provider.synthesize_stream(text)]
    data = b"".join(chunks)
    if provider.pcm_format is not None:
        return _wrap_as_wav(data, provider.pcm_format)
    return data


class FallbackTTSProvider:
    """按顺序尝试 ``providers``，第一个合成成功的候选即为最终结果。"""

    def __init__(self, providers: list[TTSProvider]) -> None:
        self._providers = providers
        self.pcm_format: PcmFormat | None = None

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        last_error: Exception | None = None
        for provider in self._providers:
            try:
                data = await _synthesize_whole(provider, text)
            except Exception as exc:
                logger.warning(
                    "TTS 候选 %s 合成失败，尝试下一个", type(provider).__name__, exc_info=exc
                )
                last_error = exc
                continue
            yield data
            return
        assert last_error is not None
        raise last_error
