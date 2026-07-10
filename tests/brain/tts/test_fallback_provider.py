"""``FallbackTTSProvider`` 的回归测试：候选顺序尝试、格式安全包装、全部失败时 raise。"""

from __future__ import annotations

import wave
from collections.abc import AsyncIterator
from io import BytesIO

import pytest

from miku_on_desk.brain.tts.base import PcmFormat, TTSProvider
from miku_on_desk.brain.tts.fallback_provider import FallbackTTSProvider


class _FakeProvider:
    def __init__(
        self,
        *,
        pcm_format: PcmFormat | None,
        chunks: list[bytes] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.pcm_format = pcm_format
        self._chunks = chunks or []
        self._error = error
        self.calls = 0

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        for chunk in self._chunks:
            yield chunk


async def _collect(provider: TTSProvider, text: str = "你好") -> bytes:
    chunks = [chunk async for chunk in provider.synthesize_stream(text)]
    assert len(chunks) == 1
    return chunks[0]


async def test_primary_success_mp3_passes_through_unchanged() -> None:
    primary = _FakeProvider(pcm_format=None, chunks=[b"id3-mp3-bytes"])
    fallback = FallbackTTSProvider([primary])

    result = await _collect(fallback)

    assert result == b"id3-mp3-bytes"


async def test_primary_success_pcm_is_wrapped_as_valid_wav() -> None:
    fmt = PcmFormat(sample_rate=24000, channels=1)
    pcm_bytes = b"\x01\x00\x02\x00\x03\x00\x04\x00"
    primary = _FakeProvider(pcm_format=fmt, chunks=[pcm_bytes])
    fallback = FallbackTTSProvider([primary])

    result = await _collect(fallback)

    with wave.open(BytesIO(result), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.readframes(wav_file.getnframes()) == pcm_bytes


async def test_primary_failure_falls_back_to_second_candidate() -> None:
    primary = _FakeProvider(pcm_format=None, error=RuntimeError("key invalid"))
    secondary = _FakeProvider(pcm_format=None, chunks=[b"edge-mp3-bytes"])
    fallback = FallbackTTSProvider([primary, secondary])

    result = await _collect(fallback)

    assert result == b"edge-mp3-bytes"
    assert primary.calls == 1
    assert secondary.calls == 1


async def test_all_candidates_failing_raises_last_error() -> None:
    primary = _FakeProvider(pcm_format=None, error=RuntimeError("primary down"))
    secondary = _FakeProvider(pcm_format=None, error=RuntimeError("edge down"))
    fallback = FallbackTTSProvider([primary, secondary])

    with pytest.raises(RuntimeError, match="edge down"):
        await _collect(fallback)


async def test_pcm_format_is_always_none() -> None:
    fallback = FallbackTTSProvider([_FakeProvider(pcm_format=PcmFormat(sample_rate=16000))])

    assert fallback.pcm_format is None
