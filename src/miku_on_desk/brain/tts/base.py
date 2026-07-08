"""TTS Provider 抽象：把一段文本合成为音频字节。

用 ``Protocol`` 而非 ABC——播放侧（``SpeechController``）只依赖"有一个 ``synthesize``
协程"这一结构，不需要继承关系，也便于测试里传入任意假实现。实现应当无状态、可被并发
调用（同一实例可能被多次 ``await``），返回完整的音频文件字节（当前实现为 mp3）。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TTSProvider(Protocol):
    async def synthesize(self, text: str) -> bytes:
        """把 ``text`` 合成为音频字节；``text`` 保证为非空、已去除首尾空白的一句话。"""
        ...
