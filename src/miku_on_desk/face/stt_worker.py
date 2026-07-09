"""语音输入后台线程：持久线程 + 单次 ``asyncio.run()`` 贯穿线程整个生命周期，跨多次录音
会话复用。

跟 ``speech_controller.py::_SynthWorker`` 同构（持久线程模式），而不是
``voice_clone_worker.py::VoiceCloneWorker`` 的一次性阻塞调用模式——WebSocket 连接在每次
录音期间是长连接、双向持续收发，跟一次性 HTTP 调用的形状不一样。

``begin_session``/``push_chunk``/``end_session``/``stop`` 从 UI 线程调用，只是往一个线程安全
的 ``queue.Queue`` 里放消息；worker 线程内部用 ``asyncio.to_thread(self._queue.get)`` 阻塞
等待——与 ``main.py`` 里 ``await asyncio.to_thread(chat_input.get)`` 是同一个"让 asyncio 循环
等待一个普通线程安全队列"的写法。四个转写信号从 worker 线程 emit，靠 Qt 默认
``AutoConnection`` 自动升级成 ``QueuedConnection`` 排队投递到 UI 线程的槽，不需要手动
``call_soon_threadsafe``。

用 ``session_id`` 而不是布尔"当前会话"来标记，是因为 UI 侧可能在上一个会话的收尾信号
（``session_closed``）到达前就已经开始了下一次录音；调用方（``ChatPopup``）据此过滤属于
已结束会话的陈旧信号。
"""

from __future__ import annotations

import asyncio
import logging
import queue
from typing import Final

from PySide6.QtCore import QObject, QThread, Signal

from miku_on_desk.brain.stt.base import STTProvider, STTSession

logger = logging.getLogger(__name__)

_BEGIN: Final = object()
_END: Final = object()
_STOP: Final = object()

_QueueItem = object | bytes


class SttWorker(QThread):
    partial_transcript = Signal(int, str)
    committed_transcript = Signal(int, str)
    session_error = Signal(int, str)
    session_closed = Signal(int)

    def __init__(self, provider: STTProvider, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._provider = provider
        self._queue: queue.Queue[tuple[int, _QueueItem]] = queue.Queue()
        self._next_session_id = 0

    def begin_session(self) -> int:
        """在 UI 线程调用：分配一个新的 session_id 并入队开始信号，立即返回该 id。"""
        self._next_session_id += 1
        session_id = self._next_session_id
        self._queue.put((session_id, _BEGIN))
        return session_id

    def push_chunk(self, session_id: int, pcm: bytes) -> None:
        self._queue.put((session_id, pcm))

    def end_session(self, session_id: int) -> None:
        self._queue.put((session_id, _END))

    def stop(self) -> None:
        """退出应用时调用：结束 worker 线程本身（与结束某次录音会话是两件事）。"""
        self._queue.put((0, _STOP))

    def run(self) -> None:
        asyncio.run(self._loop_main())

    async def _loop_main(self) -> None:
        while True:
            session_id, item = await asyncio.to_thread(self._queue.get)
            if item is _STOP:
                return
            if item is _BEGIN:
                should_stop = await self._run_session(session_id)
                if should_stop:
                    return
            # 其余情形（属于已经结束会话的孤立 chunk/_END）直接丢弃

    async def _run_session(self, session_id: int) -> bool:
        """驱动一次会话直到收到 _END/_STOP，或碰到新会话的 _BEGIN（防御性：先关旧会话再
        转交，理论上不该发生，因为 UI 侧点击切换是互斥的）。返回 True 表示应当停止整个
        worker 循环。
        """
        closing = False

        def _on_partial(text: str) -> None:
            self.partial_transcript.emit(session_id, text)

        def _on_committed(text: str) -> None:
            self.committed_transcript.emit(session_id, text)

        def _on_error(message: str) -> None:
            self.session_error.emit(session_id, message)

        def _on_close() -> None:
            if not closing:
                self.session_error.emit(session_id, "连接意外断开")
            self.session_closed.emit(session_id)

        try:
            session = await self._provider.open_session(
                on_partial=_on_partial,
                on_committed=_on_committed,
                on_error=_on_error,
                on_close=_on_close,
            )
        except Exception:
            logger.exception("开启语音输入会话失败")
            self.session_error.emit(session_id, "开启语音输入会话失败")
            self.session_closed.emit(session_id)
            return False

        while True:
            item_session_id, item = await asyncio.to_thread(self._queue.get)

            if item is _STOP:
                closing = True
                await self._safe_close(session)
                return True

            if item_session_id != session_id:
                if item is _BEGIN:
                    closing = True
                    await self._safe_close(session)
                    return await self._run_session(item_session_id)
                continue  # 属于旧会话的杂项消息，丢弃

            if item is _END:
                closing = True
                await self._safe_close(session)
                return False

            assert isinstance(item, bytes)
            try:
                await session.send_chunk(item)
            except Exception:
                logger.exception("发送语音输入音频块失败")
                closing = True
                await self._safe_close(session)
                self.session_error.emit(session_id, "发送音频失败")
                return False

    @staticmethod
    async def _safe_close(session: STTSession) -> None:
        try:
            await session.close()
        except Exception:
            logger.exception("关闭语音输入会话失败")
