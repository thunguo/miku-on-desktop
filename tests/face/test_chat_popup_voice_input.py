"""``ChatPopup`` 麦克风状态机测试：直接手动 emit ``PcmAudioCapture``/``SttWorker`` 的信号
驱动，不接真实麦克风硬件或 WebSocket。

用真实的 ``PcmAudioCapture``/``SttWorker`` 实例（它们的 Signal 是类级描述符，``ChatPopup``
构造时要 ``.connect()`` 到具体实例上，纯手搓的假对象拿不到这个），但把会触碰硬件/后台线程的
``start_capture``/``stop_capture``/``begin_session``/``end_session``/``push_chunk``
monkeypatch 成记录调用的桩。仿照 ``tests/face/test_chat_popup.py`` 现有写法。
"""

from __future__ import annotations

import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from miku_on_desk.face.stt_worker import SttWorker
from miku_on_desk.face.ui.audio_capture import PcmAudioCapture
from miku_on_desk.face.ui.chat_popup import ChatPopup, _MicState


class _InertSTTProvider:
    async def open_session(self, **_kwargs: object) -> object:  # pragma: no cover
        raise NotImplementedError


def _build_popup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ChatPopup, dict[str, list[object]]]:
    capture = PcmAudioCapture()
    worker = SttWorker(_InertSTTProvider())
    calls: dict[str, list[object]] = {
        "start_capture": [],
        "stop_capture": [],
        "begin_session": [],
        "end_session": [],
        "push_chunk": [],
    }
    monkeypatch.setattr(capture, "start_capture", lambda: calls["start_capture"].append(True))
    monkeypatch.setattr(capture, "stop_capture", lambda: calls["stop_capture"].append(True))

    def _begin_session() -> int:
        calls["begin_session"].append(True)
        return 1

    monkeypatch.setattr(worker, "begin_session", _begin_session)
    monkeypatch.setattr(worker, "end_session", lambda sid: calls["end_session"].append(sid))
    monkeypatch.setattr(
        worker, "push_chunk", lambda sid, chunk: calls["push_chunk"].append((sid, chunk))
    )
    popup = ChatPopup(voice_capture=capture, stt_worker=worker)
    return popup, calls


def test_no_mic_button_when_voice_input_not_configured(qapp: QApplication) -> None:
    popup = ChatPopup()

    assert popup._mic_button is None


def test_click_mic_button_starts_recording(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, calls = _build_popup(monkeypatch)

    popup._on_mic_button_clicked(True)

    assert popup._mic_state == _MicState.RECORDING
    assert popup._active_session_id == 1
    assert calls["begin_session"] == [True]
    assert calls["start_capture"] == [True]


def test_click_mic_button_to_start_recording_emits_barge_in_requested(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, _calls = _build_popup(monkeypatch)
    emitted: list[None] = []
    popup.barge_in_requested.connect(lambda: emitted.append(None))

    popup._on_mic_button_clicked(True)

    assert emitted == [None]


def test_click_mic_button_to_stop_recording_does_not_emit_barge_in_requested(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, _calls = _build_popup(monkeypatch)
    popup._on_mic_button_clicked(True)
    emitted: list[None] = []
    popup.barge_in_requested.connect(lambda: emitted.append(None))

    popup._on_mic_button_clicked(False)

    assert emitted == []


def test_click_mic_button_again_stops_and_begins_finalizing(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, calls = _build_popup(monkeypatch)
    popup._on_mic_button_clicked(True)

    popup._on_mic_button_clicked(False)

    assert popup._mic_state == _MicState.FINALIZING
    assert calls["stop_capture"] == [True]
    assert calls["end_session"] == [1]


def test_partial_transcript_updates_input_text_for_matching_session(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, _calls = _build_popup(monkeypatch)
    popup._on_mic_button_clicked(True)

    popup._stt_worker.partial_transcript.emit(1, "你")

    assert popup._input.text() == "你"


def test_partial_transcript_for_stale_session_is_ignored(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, _calls = _build_popup(monkeypatch)
    popup._on_mic_button_clicked(True)

    popup._stt_worker.partial_transcript.emit(999, "旧会话残留")

    assert popup._input.text() == ""


def test_committed_transcript_appends_to_prefix(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, _calls = _build_popup(monkeypatch)
    popup._on_mic_button_clicked(True)

    popup._stt_worker.committed_transcript.emit(1, "你好。")
    popup._stt_worker.partial_transcript.emit(1, "在")

    assert popup._input.text() == "你好。在"


def test_user_typing_interrupts_recording(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, calls = _build_popup(monkeypatch)
    popup._on_mic_button_clicked(True)

    popup._on_text_edited("手动打字")

    assert popup._mic_state == _MicState.IDLE
    assert calls["stop_capture"] == [True]
    assert calls["end_session"] == [1]
    assert popup._active_session_id is None


def test_session_closed_after_finalizing_returns_to_idle(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, _calls = _build_popup(monkeypatch)
    popup._on_mic_button_clicked(True)
    popup._on_mic_button_clicked(False)

    popup._stt_worker.session_closed.emit(1)

    assert popup._mic_state == _MicState.IDLE
    assert popup._active_session_id is None


def test_session_closed_for_mismatched_session_is_ignored(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, _calls = _build_popup(monkeypatch)
    popup._on_mic_button_clicked(True)

    popup._stt_worker.session_closed.emit(999)

    assert popup._mic_state == _MicState.RECORDING
    assert popup._active_session_id == 1


def test_session_error_for_matching_session_shows_error_and_stops_capture(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, calls = _build_popup(monkeypatch)
    popup._on_mic_button_clicked(True)

    popup._stt_worker.session_error.emit(1, "网络错误")

    assert popup._mic_state == _MicState.ERROR
    assert popup._input.placeholderText() == "网络错误"
    assert calls["stop_capture"] == [True]
    assert popup._active_session_id is None


def test_error_display_timeout_returns_to_idle_and_restores_placeholder(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, _calls = _build_popup(monkeypatch)
    popup._on_mic_button_clicked(True)
    popup._stt_worker.session_error.emit(1, "网络错误")

    popup._on_error_display_timeout()

    assert popup._mic_state == _MicState.IDLE
    assert popup._input.placeholderText() == "对 Miku 说点什么…"


def test_max_duration_reached_while_recording_begins_finalizing(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, calls = _build_popup(monkeypatch)
    popup._on_mic_button_clicked(True)

    popup._voice_capture.max_duration_reached.emit()

    assert popup._mic_state == _MicState.FINALIZING
    assert calls["end_session"] == [1]


def test_max_duration_reached_while_idle_is_ignored(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, calls = _build_popup(monkeypatch)

    popup._voice_capture.max_duration_reached.emit()

    assert popup._mic_state == _MicState.IDLE
    assert calls["end_session"] == []


def test_capture_unavailable_while_recording_ends_session_and_shows_error(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, calls = _build_popup(monkeypatch)
    popup._on_mic_button_clicked(True)

    popup._voice_capture.capture_unavailable.emit("未检测到可用的麦克风")

    assert calls["end_session"] == [1]
    assert popup._mic_state == _MicState.ERROR
    assert popup._input.placeholderText() == "未检测到可用的麦克风"


def test_finalize_timeout_forces_idle_when_still_finalizing(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, _calls = _build_popup(monkeypatch)
    popup._on_mic_button_clicked(True)
    popup._on_mic_button_clicked(False)

    popup._on_finalize_timeout()

    assert popup._mic_state == _MicState.IDLE


def test_chunk_captured_forwards_to_worker_while_session_active(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, calls = _build_popup(monkeypatch)
    popup._on_mic_button_clicked(True)

    popup._voice_capture.chunk_captured.emit(b"pcm-bytes")

    assert calls["push_chunk"] == [(1, b"pcm-bytes")]


def test_close_event_interrupts_active_recording(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    popup, calls = _build_popup(monkeypatch)
    popup._on_mic_button_clicked(True)

    popup.closeEvent(QCloseEvent())

    assert calls["stop_capture"] == [True]
    assert calls["end_session"] == [1]
    assert popup._mic_state == _MicState.IDLE
