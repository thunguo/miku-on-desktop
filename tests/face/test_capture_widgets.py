"""``capture_widgets`` 的纯逻辑分支与可注入权限回调测试。

真实拍照/录音成功路径需要真实摄像头/麦克风硬件与操作系统权限授权，不进 CI，属于
计划里标注的人工验证项。这里只覆盖：设备探测的纯逻辑分支、权限三态（Denied/
Undetermined/Granted）路由是否调用了正确的下一步——通过 monkeypatch 掉真正触碰硬件的
``_start_session``/``_start_recording``，验证路由逻辑而不构造真实 ``QCamera``/``QAudioInput``。
"""

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from miku_on_desk.face.ui import capture_widgets
from miku_on_desk.face.ui.capture_widgets import (
    AudioRecorderWidget,
    CameraCaptureWidget,
    probe_capture_availability,
)


def test_probe_capture_availability_reports_both_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        capture_widgets.QMediaDevices, "videoInputs", staticmethod(lambda: [object()])
    )
    monkeypatch.setattr(
        capture_widgets.QMediaDevices, "audioInputs", staticmethod(lambda: [object()])
    )

    assert probe_capture_availability() == (True, True)


def test_probe_capture_availability_reports_both_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capture_widgets.QMediaDevices, "videoInputs", staticmethod(lambda: []))
    monkeypatch.setattr(capture_widgets.QMediaDevices, "audioInputs", staticmethod(lambda: []))

    assert probe_capture_availability() == (False, False)


def _no_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capture_widgets.QMediaDevices, "videoInputs", staticmethod(lambda: []))
    monkeypatch.setattr(
        capture_widgets.QMediaDevices, "audioInputs", staticmethod(lambda: [object()])
    )


def _no_microphone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        capture_widgets.QMediaDevices, "videoInputs", staticmethod(lambda: [object()])
    )
    monkeypatch.setattr(capture_widgets.QMediaDevices, "audioInputs", staticmethod(lambda: []))


def test_camera_capture_widget_start_emits_unavailable_when_no_camera_device(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_camera(monkeypatch)
    widget = CameraCaptureWidget(check_permission=lambda: Qt.PermissionStatus.Granted)
    messages: list[str] = []
    widget.capture_unavailable.connect(messages.append)

    widget.start()

    assert messages == ["未检测到可用的摄像头"]


def test_camera_capture_widget_start_emits_unavailable_when_permission_denied(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        capture_widgets.QMediaDevices, "videoInputs", staticmethod(lambda: [object()])
    )
    widget = CameraCaptureWidget(check_permission=lambda: Qt.PermissionStatus.Denied)
    messages: list[str] = []
    widget.capture_unavailable.connect(messages.append)

    widget.start()

    assert len(messages) == 1


def test_camera_capture_widget_start_requests_permission_when_undetermined(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        capture_widgets.QMediaDevices, "videoInputs", staticmethod(lambda: [object()])
    )
    requested: list[Any] = []
    widget = CameraCaptureWidget(
        check_permission=lambda: Qt.PermissionStatus.Undetermined,
        request_permission=lambda callback: requested.append(callback),
    )

    widget.start()

    assert len(requested) == 1


def test_camera_capture_widget_permission_result_denied_emits_unavailable(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        capture_widgets.QMediaDevices, "videoInputs", staticmethod(lambda: [object()])
    )
    widget = CameraCaptureWidget(check_permission=lambda: Qt.PermissionStatus.Granted)
    messages: list[str] = []
    widget.capture_unavailable.connect(messages.append)

    widget._on_permission_result(Qt.PermissionStatus.Denied)

    assert len(messages) == 1


def test_camera_capture_widget_start_calls_start_session_when_granted(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        capture_widgets.QMediaDevices, "videoInputs", staticmethod(lambda: [object()])
    )
    widget = CameraCaptureWidget(check_permission=lambda: Qt.PermissionStatus.Granted)
    started: list[bool] = []
    monkeypatch.setattr(widget, "_start_session", lambda: started.append(True))

    widget.start()

    assert started == [True]


def test_camera_capture_widget_capture_photo_emits_unavailable_without_session(
    qapp: QApplication,
) -> None:
    widget = CameraCaptureWidget()
    messages: list[str] = []
    widget.capture_unavailable.connect(messages.append)

    widget.capture_photo()

    assert len(messages) == 1


def test_audio_recorder_widget_start_emits_unavailable_when_no_microphone_device(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_microphone(monkeypatch)
    widget = AudioRecorderWidget(check_permission=lambda: Qt.PermissionStatus.Granted)
    messages: list[str] = []
    widget.recording_unavailable.connect(messages.append)

    widget.start()

    assert messages == ["未检测到可用的麦克风"]


def test_audio_recorder_widget_start_emits_unavailable_when_permission_denied(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        capture_widgets.QMediaDevices, "audioInputs", staticmethod(lambda: [object()])
    )
    widget = AudioRecorderWidget(check_permission=lambda: Qt.PermissionStatus.Denied)
    messages: list[str] = []
    widget.recording_unavailable.connect(messages.append)

    widget.start()

    assert len(messages) == 1


def test_audio_recorder_widget_start_requests_permission_when_undetermined(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        capture_widgets.QMediaDevices, "audioInputs", staticmethod(lambda: [object()])
    )
    requested: list[Any] = []
    widget = AudioRecorderWidget(
        check_permission=lambda: Qt.PermissionStatus.Undetermined,
        request_permission=lambda callback: requested.append(callback),
    )

    widget.start()

    assert len(requested) == 1


def test_audio_recorder_widget_start_calls_start_recording_when_granted(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        capture_widgets.QMediaDevices, "audioInputs", staticmethod(lambda: [object()])
    )
    widget = AudioRecorderWidget(check_permission=lambda: Qt.PermissionStatus.Granted)
    started: list[bool] = []
    monkeypatch.setattr(widget, "_start_recording", lambda: started.append(True))

    widget.start()

    assert started == [True]


def test_audio_recorder_widget_finalize_recording_emits_unavailable_when_no_recorder(
    qapp: QApplication,
) -> None:
    widget = AudioRecorderWidget()
    messages: list[str] = []
    widget.recording_unavailable.connect(messages.append)

    widget._finalize_recording()

    assert len(messages) == 1


def test_audio_recorder_widget_recorder_error_emits_unavailable_once(
    qapp: QApplication,
) -> None:
    widget = AudioRecorderWidget()
    messages: list[str] = []
    widget.recording_unavailable.connect(messages.append)

    widget._on_recorder_error(capture_widgets.QMediaRecorder.Error.ResourceError, "设备被占用")
    widget._on_recorder_error(capture_widgets.QMediaRecorder.Error.ResourceError, "设备被占用")

    assert messages == ["设备被占用"]
