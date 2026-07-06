"""流式响应的双超时看门狗：idle timeout 检测"流已死"，hard timeout 是绝对兜底上限。

单一超时无法区分"连接已死"和"模型正在思考 30-60 秒但还没有任何可见事件"，所以拆成两个独立
计时器——idle 计时器在每次收到流事件时重置，只在真正沉默时触发；hard 计时器是绝对墙钟上限，
不受活动影响。一个促成 idle timeout 存在的真实故障模式：上游把超大 tool_use 负载截断后连接
会陷入沉默但不断开，与其等上游自己的几分钟级空闲超时，不如更快地主动放弃并给出"payload 太大，
拆分重试"这样可操作的错误，而不是让用户对着卡住的对话干等。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator


class StreamTimeoutError(Exception):
    def __init__(self, kind: str) -> None:
        self.kind = kind  # "idle" 或 "hard"
        super().__init__(f"stream {kind} timeout")


async def watch_stream_timeouts[T](
    events: AsyncIterator[T],
    *,
    idle_timeout_s: float,
    hard_timeout_s: float,
) -> AsyncIterator[T]:
    """包一层双超时看门狗；任一超时触发时向调用方抛出 ``StreamTimeoutError``。"""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + hard_timeout_s
    iterator = events.__aiter__()
    while True:
        remaining_hard = deadline - loop.time()
        if remaining_hard <= 0:
            raise StreamTimeoutError("hard")
        timeout = min(idle_timeout_s, remaining_hard)
        try:
            item = await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
        except TimeoutError:
            if loop.time() >= deadline:
                raise StreamTimeoutError("hard") from None
            raise StreamTimeoutError("idle") from None
        except StopAsyncIteration:
            return
        yield item
