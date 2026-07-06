"""watch_stream_timeouts 双超时看门狗的回归测试：idle 计时器重置、hard 计时器绝对上限。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from miku_on_desk.brain.providers.stream_timeout import StreamTimeoutError, watch_stream_timeouts


async def _emit(items: list[int], delay: float = 0.0) -> AsyncIterator[int]:
    for item in items:
        if delay:
            await asyncio.sleep(delay)
        yield item


async def _collect(events: AsyncIterator[int]) -> list[int]:
    return [item async for item in events]


async def test_all_events_pass_through_when_no_timeout_triggered() -> None:
    result = await _collect(
        watch_stream_timeouts(_emit([1, 2, 3]), idle_timeout_s=1.0, hard_timeout_s=1.0)
    )
    assert result == [1, 2, 3]


async def test_idle_timeout_raised_when_gap_between_events_too_long() -> None:
    async def gen() -> AsyncIterator[int]:
        yield 1
        await asyncio.sleep(0.2)
        yield 2

    with pytest.raises(StreamTimeoutError) as exc_info:
        await _collect(watch_stream_timeouts(gen(), idle_timeout_s=0.05, hard_timeout_s=10.0))
    assert exc_info.value.kind == "idle"


async def test_hard_timeout_raised_even_with_frequent_activity() -> None:
    async def gen() -> AsyncIterator[int]:
        for _ in range(20):
            await asyncio.sleep(0.02)
            yield 1

    with pytest.raises(StreamTimeoutError) as exc_info:
        await _collect(watch_stream_timeouts(gen(), idle_timeout_s=1.0, hard_timeout_s=0.05))
    assert exc_info.value.kind == "hard"


async def test_idle_timer_resets_on_each_event() -> None:
    async def gen() -> AsyncIterator[int]:
        for _ in range(5):
            await asyncio.sleep(0.03)
            yield 1

    result = await _collect(
        watch_stream_timeouts(gen(), idle_timeout_s=0.05, hard_timeout_s=10.0)
    )
    assert result == [1, 1, 1, 1, 1]
