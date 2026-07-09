"""``SttWorker`` 的会话/队列逻辑回归测试：用假 ``STTProvider``/``STTSession`` 驱动
``_loop_main``，同步调用回调，不真的起线程、不真的开 WebSocket。

不调用 ``SttWorker.start()``（不起真正的 ``QThread``），直接用 ``asyncio.run`` 在测试线程
里跑 ``_loop_main``——测试前把所有输入通过公开方法一次性放进队列，``_loop_main`` 处理完
排到的 ``_STOP`` 后自然返回，不需要真正的后台线程。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from PySide6.QtWidgets import QApplication

from miku_on_desk.face.stt_worker import SttWorker


class _FakeSTTSession:
    """``close()`` 同步触发 ``on_close``，对应真实 SDK「关闭请求最终会让服务端触发 CLOSE
    事件」的契约。
    """

    def __init__(self, on_close: Callable[[], None]) -> None:
        self.sent_chunks: list[bytes] = []
        self.closed = False
        self._on_close = on_close

    async def send_chunk(self, pcm: bytes) -> None:
        self.sent_chunks.append(pcm)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._on_close()


class _FakeSTTProvider:
    """``open_session`` 返回前可选地同步触发 partial/committed/error/close 回调，模拟真实
    SDK 的事件时序。
    """

    def __init__(
        self,
        *,
        partial_text: str | None = None,
        committed_text: str | None = None,
        error_message: str | None = None,
        disconnect_immediately: bool = False,
    ) -> None:
        self.sessions: list[_FakeSTTSession] = []
        self._partial_text = partial_text
        self._committed_text = committed_text
        self._error_message = error_message
        self._disconnect_immediately = disconnect_immediately

    async def open_session(
        self,
        *,
        on_partial: Callable[[str], None],
        on_committed: Callable[[str], None],
        on_error: Callable[[str], None],
        on_close: Callable[[], None],
    ) -> _FakeSTTSession:
        session = _FakeSTTSession(on_close)
        self.sessions.append(session)
        if self._partial_text is not None:
            on_partial(self._partial_text)
        if self._committed_text is not None:
            on_committed(self._committed_text)
        if self._error_message is not None:
            on_error(self._error_message)
        if self._disconnect_immediately:
            # 走 session.close() 而不是直接调用 on_close()，这样 _FakeSTTSession 的幂等
            # 守卫会生效——真实 SDK 的连接在已关闭后不会为后续的 close() 调用重复触发一次
            # CLOSE 事件，测试替身要如实模拟这一点，否则后面 _STOP 触发的 _safe_close 会
            # 让 on_close 被再喊一次。
            await session.close()
        return session


def test_begin_session_returns_incrementing_session_ids(qapp: QApplication) -> None:
    worker = SttWorker(_FakeSTTProvider())

    assert worker.begin_session() == 1
    assert worker.begin_session() == 2


def test_full_session_lifecycle_sends_chunks_and_closes_without_error(
    qapp: QApplication,
) -> None:
    provider = _FakeSTTProvider()
    worker = SttWorker(provider)
    errors: list[tuple[int, str]] = []
    closed: list[int] = []
    worker.session_error.connect(lambda sid, msg: errors.append((sid, msg)))
    worker.session_closed.connect(closed.append)

    session_id = worker.begin_session()
    worker.push_chunk(session_id, b"pcm-bytes")
    worker.end_session(session_id)
    worker.stop()
    asyncio.run(worker._loop_main())

    assert provider.sessions[0].sent_chunks == [b"pcm-bytes"]
    assert provider.sessions[0].closed is True
    assert errors == []
    assert closed == [session_id]


def test_provider_relays_partial_and_committed_transcripts_with_session_id(
    qapp: QApplication,
) -> None:
    provider = _FakeSTTProvider(partial_text="你", committed_text="你好。")
    worker = SttWorker(provider)
    partials: list[tuple[int, str]] = []
    committed: list[tuple[int, str]] = []
    worker.partial_transcript.connect(lambda sid, text: partials.append((sid, text)))
    worker.committed_transcript.connect(lambda sid, text: committed.append((sid, text)))

    session_id = worker.begin_session()
    worker.end_session(session_id)
    worker.stop()
    asyncio.run(worker._loop_main())

    assert partials == [(session_id, "你")]
    assert committed == [(session_id, "你好。")]


def test_provider_error_callback_emits_session_error_with_session_id(
    qapp: QApplication,
) -> None:
    provider = _FakeSTTProvider(error_message="quota exceeded")
    worker = SttWorker(provider)
    errors: list[tuple[int, str]] = []
    worker.session_error.connect(lambda sid, msg: errors.append((sid, msg)))

    session_id = worker.begin_session()
    worker.stop()
    asyncio.run(worker._loop_main())

    assert errors == [(session_id, "quota exceeded")]


def test_unexpected_disconnect_before_end_emits_error_then_closed(
    qapp: QApplication,
) -> None:
    provider = _FakeSTTProvider(disconnect_immediately=True)
    worker = SttWorker(provider)
    errors: list[tuple[int, str]] = []
    closed: list[int] = []
    worker.session_error.connect(lambda sid, msg: errors.append((sid, msg)))
    worker.session_closed.connect(closed.append)

    session_id = worker.begin_session()
    worker.stop()
    asyncio.run(worker._loop_main())

    assert errors == [(session_id, "连接意外断开")]
    assert closed == [session_id]


def test_new_begin_session_closes_previous_session_without_emitting_error(
    qapp: QApplication,
) -> None:
    provider = _FakeSTTProvider()
    worker = SttWorker(provider)
    errors: list[tuple[int, str]] = []
    closed: list[int] = []
    worker.session_error.connect(lambda sid, msg: errors.append((sid, msg)))
    worker.session_closed.connect(closed.append)

    first_id = worker.begin_session()
    second_id = worker.begin_session()
    worker.end_session(second_id)
    worker.stop()
    asyncio.run(worker._loop_main())

    assert provider.sessions[0].closed is True
    assert provider.sessions[1].closed is True
    assert errors == []
    assert closed == [first_id, second_id]


def test_stale_chunk_for_already_ended_session_is_discarded(qapp: QApplication) -> None:
    provider = _FakeSTTProvider()
    worker = SttWorker(provider)

    session_id = worker.begin_session()
    worker.end_session(session_id)
    worker.push_chunk(session_id, b"too-late")
    worker.stop()
    asyncio.run(worker._loop_main())

    assert provider.sessions[0].sent_chunks == []
