"""``PcmAudioCapture`` 的权限分支/纯逻辑测试。

真实 ``QAudioSource`` 硬件采集需要真实麦克风与系统权限授权，不进 CI，属于计划里标注的
人工验证项。这里只覆盖：设备探测降级、权限三态（Denied/Undetermined/Granted）路由是否调用了
正确的下一步、``readyRead``/状态变化/最长时长兜底的纯逻辑分支——通过 monkeypatch 掉真正触碰
硬件的 ``_start_source``，或直接注入假的 source/device，不构造真实 ``QAudioSource``。仿照
``tests/face/test_capture_widgets.py``。
"""

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtMultimedia import QAudio, QtAudio
from PySide6.QtWidgets import QApplication

from miku_on_desk.face.ui import audio_capture
from miku_on_desk.face.ui.audio_capture import PcmAudioCapture


class _FakeIoDevice:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def readAll(self) -> _FakeIoDevice:
        return self

    def data(self) -> bytes:
        return self._data


class _FakeSource:
    def __init__(self, *, error: QtAudio.Error = QtAudio.Error.NoError) -> None:
        self.stopped = False
        self._error = error

    def stop(self) -> None:
        self.stopped = True

    def error(self) -> QtAudio.Error:
        return self._error


def _has_microphone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        audio_capture.QMediaDevices, "audioInputs", staticmethod(lambda: [object()])
    )


def _no_microphone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_capture.QMediaDevices, "audioInputs", staticmethod(lambda: []))


def test_start_capture_emits_unavailable_when_no_microphone_device(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_microphone(monkeypatch)
    capture = PcmAudioCapture(check_permission=lambda: Qt.PermissionStatus.Granted)
    messages: list[str] = []
    capture.capture_unavailable.connect(messages.append)

    capture.start_capture()

    assert messages == ["未检测到可用的麦克风"]


def test_start_capture_emits_unavailable_when_permission_denied(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _has_microphone(monkeypatch)
    capture = PcmAudioCapture(check_permission=lambda: Qt.PermissionStatus.Denied)
    messages: list[str] = []
    capture.capture_unavailable.connect(messages.append)

    capture.start_capture()

    assert len(messages) == 1


def test_start_capture_requests_permission_when_undetermined(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _has_microphone(monkeypatch)
    requested: list[Any] = []
    capture = PcmAudioCapture(
        check_permission=lambda: Qt.PermissionStatus.Undetermined,
        request_permission=lambda callback: requested.append(callback),
    )

    capture.start_capture()

    assert len(requested) == 1


def test_permission_result_denied_emits_unavailable(qapp: QApplication) -> None:
    capture = PcmAudioCapture(check_permission=lambda: Qt.PermissionStatus.Granted)
    messages: list[str] = []
    capture.capture_unavailable.connect(messages.append)

    capture._on_permission_result(Qt.PermissionStatus.Denied)

    assert len(messages) == 1


def test_start_capture_calls_start_source_when_granted(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _has_microphone(monkeypatch)
    capture = PcmAudioCapture(check_permission=lambda: Qt.PermissionStatus.Granted)
    started: list[bool] = []
    monkeypatch.setattr(capture, "_start_source", lambda: started.append(True))

    capture.start_capture()

    assert started == [True]


def test_permission_result_granted_calls_start_source(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = PcmAudioCapture()
    started: list[bool] = []
    monkeypatch.setattr(capture, "_start_source", lambda: started.append(True))

    capture._on_permission_result(Qt.PermissionStatus.Granted)

    assert started == [True]


def test_on_ready_read_emits_chunk_captured_with_available_bytes(qapp: QApplication) -> None:
    capture = PcmAudioCapture()
    chunks: list[bytes] = []
    capture.chunk_captured.connect(chunks.append)
    capture._device = _FakeIoDevice(b"pcm-bytes")  # type: ignore[assignment]

    capture._on_ready_read()

    assert chunks == [b"pcm-bytes"]


def test_on_ready_read_without_device_emits_nothing(qapp: QApplication) -> None:
    capture = PcmAudioCapture()
    chunks: list[bytes] = []
    capture.chunk_captured.connect(chunks.append)

    capture._on_ready_read()

    assert chunks == []


def test_on_state_changed_ignores_non_stopped_state(qapp: QApplication) -> None:
    capture = PcmAudioCapture()
    messages: list[str] = []
    capture.capture_unavailable.connect(messages.append)

    capture._on_state_changed(QAudio.State.ActiveState)

    assert messages == []


def test_on_state_changed_stopped_with_error_emits_unavailable_and_cleans_up(
    qapp: QApplication,
) -> None:
    capture = PcmAudioCapture()
    capture._source = _FakeSource(error=QtAudio.Error.FatalError)  # type: ignore[assignment]
    capture._device = _FakeIoDevice(b"")  # type: ignore[assignment]
    messages: list[str] = []
    capture.capture_unavailable.connect(messages.append)

    capture._on_state_changed(QAudio.State.StoppedState)

    assert messages == ["麦克风采集异常，请重试"]
    assert capture._source is None
    assert capture._device is None


def test_on_state_changed_stopped_without_error_does_not_clean_up(qapp: QApplication) -> None:
    capture = PcmAudioCapture()
    source = _FakeSource(error=QtAudio.Error.NoError)
    capture._source = source  # type: ignore[assignment]
    messages: list[str] = []
    capture.capture_unavailable.connect(messages.append)

    capture._on_state_changed(QAudio.State.StoppedState)

    assert messages == []
    assert capture._source is source


def test_max_duration_reached_stops_capture_and_emits_signal(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = PcmAudioCapture()
    stopped: list[bool] = []
    monkeypatch.setattr(capture, "stop_capture", lambda: stopped.append(True))
    reached: list[bool] = []
    capture.max_duration_reached.connect(lambda: reached.append(True))

    capture._on_max_duration_reached()

    assert stopped == [True]
    assert reached == [True]


def test_stop_capture_stops_existing_source_and_cleans_up(qapp: QApplication) -> None:
    capture = PcmAudioCapture()
    source = _FakeSource()
    capture._source = source  # type: ignore[assignment]
    capture._device = _FakeIoDevice(b"")  # type: ignore[assignment]

    capture.stop_capture()

    assert source.stopped is True
    assert capture._source is None
    assert capture._device is None


def test_stop_capture_without_source_does_not_raise(qapp: QApplication) -> None:
    capture = PcmAudioCapture()

    capture.stop_capture()

    assert capture._source is None
    assert capture._device is None
