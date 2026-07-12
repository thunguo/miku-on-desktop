"""摄像头拍照 + 麦克风录音组件。

拍照和录音是顺序发生的两个独立会话（各自持有自己的 ``QMediaCaptureSession``），
不共用同一个会话——同一 session 挂 camera+recorder 会连视频一起录下来，不是这里要的效果。

权限检查/请求通过构造参数注入（``check_permission``/``request_permission``），默认实现
才落到真实的 ``QCoreApplication.checkPermission``/``requestPermission``。这样测试可以在
不触碰 Qt 静态权限 API 的前提下覆盖 Denied/Granted/Undetermined 三条分支。
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import (
    QBuffer,
    QCameraPermission,
    QCoreApplication,
    QIODeviceBase,
    QMicrophonePermission,
    QObject,
    QPermission,
    Qt,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import QImage
from PySide6.QtMultimedia import (
    QAudioInput,
    QCamera,
    QImageCapture,
    QMediaCaptureSession,
    QMediaDevices,
    QMediaFormat,
    QMediaRecorder,
)
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QVBoxLayout, QWidget

from miku_on_desk.face.ui.theme import RADIUS_MD, TEAL_DARK, border_qss
from miku_on_desk.hardware.video import HardwareCaptureError, StillCameraSource

logger = logging.getLogger(__name__)

_RECORDING_DURATION_MS = 30_000
_MIN_RECORDING_BYTES = 1024


def probe_capture_availability() -> tuple[bool, bool]:
    """(camera_available, microphone_available)。

    在真正构造 ``QCamera``/``QAudioInput`` 之前调用，用于提前展示"未检测到摄像头/麦克风"
    的降级文案。
    """
    return bool(QMediaDevices.videoInputs()), bool(QMediaDevices.audioInputs())


class _PermissionResultRelay(QObject):
    """``requestPermission`` 的第三个参数必须是绑定方法（要有 ``__func__``/``__self__``），
    传普通函数或闭包会在 PySide6 绑定层报 ``AttributeError``，所以用这个 QObject 包一层。
    """

    def __init__(
        self, callback: Callable[[Qt.PermissionStatus], None], parent: QObject
    ) -> None:
        super().__init__(parent)
        self._callback = callback

    def on_result(self, permission: QPermission) -> None:
        self._callback(permission.status())


def _default_check_camera_permission() -> Qt.PermissionStatus:
    app = QCoreApplication.instance()
    assert app is not None
    return app.checkPermission(QCameraPermission())


def _default_request_camera_permission(callback: Callable[[Qt.PermissionStatus], None]) -> None:
    app = QCoreApplication.instance()
    assert app is not None
    relay = _PermissionResultRelay(callback, app)
    app.requestPermission(QCameraPermission(), app, relay.on_result)


def default_check_microphone_permission() -> Qt.PermissionStatus:
    app = QCoreApplication.instance()
    assert app is not None
    return app.checkPermission(QMicrophonePermission())


def default_request_microphone_permission(
    callback: Callable[[Qt.PermissionStatus], None],
) -> None:
    app = QCoreApplication.instance()
    assert app is not None
    relay = _PermissionResultRelay(callback, app)
    app.requestPermission(QMicrophonePermission(), app, relay.on_result)


class CameraCaptureWidget(QWidget):
    """内嵌取景框，拍一张照片并以 PNG bytes 回传。"""

    photo_captured = Signal(bytes)
    capture_unavailable = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        still_source: StillCameraSource | None = None,
        check_permission: Callable[[], Qt.PermissionStatus] | None = None,
        request_permission: Callable[[Callable[[Qt.PermissionStatus], None]], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._still_source = still_source
        self._check_permission = check_permission or _default_check_camera_permission
        self._request_permission = request_permission or _default_request_camera_permission
        self._camera: QCamera | None = None
        self._capture_session: QMediaCaptureSession | None = None
        self._image_capture: QImageCapture | None = None

        self._video_widget = QVideoWidget(self)
        self._video_widget.setStyleSheet(border_qss(TEAL_DARK, radius=RADIUS_MD))
        self._video_widget.setMinimumSize(320, 240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._video_widget)

    def start(self) -> None:
        if self._still_source is not None:
            if not self._still_source.is_available():
                self.capture_unavailable.emit("未检测到可用的 CSI 摄像头")
            return
        camera_available, _ = probe_capture_availability()
        if not camera_available:
            self.capture_unavailable.emit("未检测到可用的摄像头")
            return

        status = self._check_permission()
        if status == Qt.PermissionStatus.Denied:
            self.capture_unavailable.emit("摄像头权限被拒绝，请在系统设置中允许后重试")
        elif status == Qt.PermissionStatus.Undetermined:
            self._request_permission(self._on_permission_result)
        else:
            self._start_session()

    def _on_permission_result(self, status: Qt.PermissionStatus) -> None:
        if status != Qt.PermissionStatus.Granted:
            self.capture_unavailable.emit("摄像头权限被拒绝，请在系统设置中允许后重试")
            return
        self._start_session()

    def _start_session(self) -> None:
        try:
            camera = QCamera(QMediaDevices.defaultVideoInput(), self)
            camera.errorOccurred.connect(self._on_camera_error)

            image_capture = QImageCapture(self)
            image_capture.imageCaptured.connect(self._on_image_captured)
            image_capture.errorOccurred.connect(self._on_capture_error)

            session = QMediaCaptureSession(self)
            session.setCamera(camera)
            session.setImageCapture(image_capture)
            session.setVideoOutput(self._video_widget)

            self._camera = camera
            self._capture_session = session
            self._image_capture = image_capture
            camera.start()
        except Exception:
            logger.exception("启动摄像头会话失败")
            self.capture_unavailable.emit("启动摄像头失败")

    def capture_photo(self) -> None:
        if self._still_source is not None:
            try:
                self.photo_captured.emit(self._still_source.capture_png())
            except HardwareCaptureError as exc:
                logger.warning("CSI 拍照失败：%s", exc)
                self.capture_unavailable.emit(str(exc))
            return
        if self._image_capture is None:
            self.capture_unavailable.emit("摄像头未就绪")
            return
        self._image_capture.capture()

    def _on_image_captured(self, _id: int, image: QImage) -> None:
        buffer = QBuffer()
        buffer.open(QIODeviceBase.OpenModeFlag.WriteOnly)
        # PySide6 6.11 的类型标注写的是 bytes，但运行时实际只接受 str，传 bytes 会在绑定层报
        # ValueError（已用 uv run python 直接验证过），这里的 type: ignore 是绕过标注和实现不一致。
        image.save(buffer, "PNG")  # type: ignore[call-overload]
        self.photo_captured.emit(bytes(buffer.data().data()))
        buffer.close()

    def _on_capture_error(self, _id: int, _error: QImageCapture.Error, message: str) -> None:
        logger.warning("拍照失败：%s", message)
        self.capture_unavailable.emit(message or "拍照失败")

    def _on_camera_error(self, _error: QCamera.Error, message: str) -> None:
        logger.warning("摄像头错误：%s", message)
        self.capture_unavailable.emit(message or "摄像头错误")

    def stop(self) -> None:
        if self._camera is not None:
            self._camera.stop()
        self._camera = None
        self._capture_session = None
        self._image_capture = None


class AudioRecorderWidget(QWidget):
    """录制最长 30 秒的 WAV 音频并以 bytes 回传。"""

    recording_finished = Signal(bytes)
    recording_unavailable = Signal(str)
    seconds_remaining_changed = Signal(int)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        check_permission: Callable[[], Qt.PermissionStatus] | None = None,
        request_permission: Callable[[Callable[[Qt.PermissionStatus], None]], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._check_permission = check_permission or default_check_microphone_permission
        self._request_permission = request_permission or default_request_microphone_permission
        self._audio_input: QAudioInput | None = None
        self._capture_session: QMediaCaptureSession | None = None
        self._recorder: QMediaRecorder | None = None
        self._output_path: Path | None = None
        self._finished = False

        self._stop_timer = QTimer(self)
        self._stop_timer.setSingleShot(True)
        self._stop_timer.timeout.connect(self.stop)

        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(1000)
        self._seconds_remaining = _RECORDING_DURATION_MS // 1000
        self._tick_timer.timeout.connect(self._on_tick)

    def start(self) -> None:
        _, microphone_available = probe_capture_availability()
        if not microphone_available:
            self.recording_unavailable.emit("未检测到可用的麦克风")
            return

        status = self._check_permission()
        if status == Qt.PermissionStatus.Denied:
            self.recording_unavailable.emit("麦克风权限被拒绝，请在系统设置中允许后重试")
        elif status == Qt.PermissionStatus.Undetermined:
            self._request_permission(self._on_permission_result)
        else:
            self._start_recording()

    def _on_permission_result(self, status: Qt.PermissionStatus) -> None:
        if status != Qt.PermissionStatus.Granted:
            self.recording_unavailable.emit("麦克风权限被拒绝，请在系统设置中允许后重试")
            return
        self._start_recording()

    def _start_recording(self) -> None:
        self._finished = False
        try:
            fd, path_str = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            output_path = Path(path_str)
            output_path.unlink()  # QMediaRecorder 要求目标文件不存在，只借用临时文件名

            media_format = QMediaFormat()
            media_format.setFileFormat(QMediaFormat.FileFormat.Wave)
            media_format.setAudioCodec(QMediaFormat.AudioCodec.Wave)

            audio_input = QAudioInput(QMediaDevices.defaultAudioInput(), self)

            recorder = QMediaRecorder(self)
            recorder.setMediaFormat(media_format)
            recorder.setOutputLocation(QUrl.fromLocalFile(str(output_path)))
            recorder.recorderStateChanged.connect(self._on_recorder_state_changed)
            recorder.errorOccurred.connect(self._on_recorder_error)

            session = QMediaCaptureSession(self)
            session.setAudioInput(audio_input)
            session.setRecorder(recorder)

            self._audio_input = audio_input
            self._capture_session = session
            self._recorder = recorder
            self._output_path = output_path

            recorder.record()
            self._seconds_remaining = _RECORDING_DURATION_MS // 1000
            self.seconds_remaining_changed.emit(self._seconds_remaining)
            self._stop_timer.start(_RECORDING_DURATION_MS)
            self._tick_timer.start()
        except Exception:
            logger.exception("启动录音会话失败")
            self._finished = True
            self.recording_unavailable.emit("启动录音失败")

    def _on_tick(self) -> None:
        self._seconds_remaining = max(0, self._seconds_remaining - 1)
        self.seconds_remaining_changed.emit(self._seconds_remaining)

    def stop(self) -> None:
        self._stop_timer.stop()
        self._tick_timer.stop()
        if self._recorder is not None:
            self._recorder.stop()

    def _on_recorder_state_changed(self, state: QMediaRecorder.RecorderState) -> None:
        if state != QMediaRecorder.RecorderState.StoppedState or self._finished:
            return
        self._finished = True
        self._finalize_recording()

    def _on_recorder_error(self, _error: QMediaRecorder.Error, message: str) -> None:
        if self._finished:
            return
        self._finished = True
        logger.warning("录音错误：%s", message)
        self._stop_timer.stop()
        self._tick_timer.stop()
        self._cleanup_output_file()
        self.recording_unavailable.emit(message or "录音失败")

    def _finalize_recording(self) -> None:
        recorder = self._recorder
        output_path = self._output_path
        if (
            recorder is not None
            and recorder.error() == QMediaRecorder.Error.NoError
            and output_path is not None
            and output_path.exists()
            and output_path.stat().st_size >= _MIN_RECORDING_BYTES
        ):
            data = output_path.read_bytes()
            output_path.unlink(missing_ok=True)
            self.recording_finished.emit(data)
        else:
            self._cleanup_output_file()
            self.recording_unavailable.emit("录音结果异常，请重试")

    def _cleanup_output_file(self) -> None:
        if self._output_path is not None:
            self._output_path.unlink(missing_ok=True)
