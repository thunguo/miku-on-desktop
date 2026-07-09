"""TTS Provider 抽象：把一段文本合成为音频字节流。

用 ``Protocol`` 而非 ABC——播放侧（``SpeechController``）只依赖"有一个
``synthesize_stream`` 方法"这一结构，不需要继承关系，也便于测试里传入任意假实现。
实现应当无状态、可被并发调用（同一实例可能被多次调用），逐块产出音频字节；批量式引擎
（拿到完整结果才返回）只需在拿到全部字节后 yield 一次。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class PcmFormat:
    """chunk 若是裸 PCM 样本（16-bit 有符号小端），用它描述采样参数。"""

    sample_rate: int
    channels: int = 1


@runtime_checkable
class TTSProvider(Protocol):
    pcm_format: PcmFormat | None
    """``None`` 表示 chunk 是压缩容器（如 mp3）分片，播放侧退回整句缓冲 + ``QMediaPlayer``；
    非 ``None`` 表示每个 chunk 都是该格式的连续原始 PCM 样本，播放侧可直接喂 ``QAudioSink``。
    """

    def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """把 ``text``（非空、已去除首尾空白的一句话）合成为音频字节流，逐块产出。

        实现写成 ``async def`` + ``yield``（异步生成器函数）即满足本协议——调用本身不
        阻塞、不需要 ``await``，直接 ``async for`` 消费。
        """
        ...
