"""麦克风 PCM 采集：用 ``QAudioSource`` 增量捕获裸 PCM 音频供实时转写使用。

跟 ``AudioRecorderWidget``（同目录 ``capture_widgets.py``）的区别：那个组件用
``QAudioInput`` + ``QMediaRecorder`` 把整段录音写成一个 WAV 文件，录完才能拿到数据，不适合
"边说边转写"的流式场景。这里改用 ``QAudioSource`` 的 pull 模式——``start()`` 返回一个内部
``QIODevice``，通过它的 ``readyRead`` 信号增量读出裸 PCM 字节，边采集边通过
``chunk_captured`` 信号交给上层送进 STT 会话。

采样格式固定 16kHz/mono/Int16，与 ``ElevenLabsSTTProvider`` 里
``AudioFormat.PCM_16000``/``sample_rate=16000`` 对齐——采集端和转写端的采样率必须一致。

权限检查/请求的注入模式、``probe_capture_availability()`` 的复用方式，均与
``AudioRecorderWidget`` 保持一致，直接复用 ``capture_widgets.py`` 里的公开默认实现。
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QIODevice, QObject, Qt, QTimer, Signal
from PySide6.QtMultimedia import QAudio, QAudioFormat, QAudioSource, QMediaDevices, QtAudio

from miku_on_desk.face.ui.capture_widgets import (
    default_check_microphone_permission,
    default_request_microphone_permission,
    probe_capture_availability,
)

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_CHANNEL_COUNT = 1


def _pcm_audio_format() -> QAudioFormat:
    audio_format = QAudioFormat()
    audio_format.setSampleRate(_SAMPLE_RATE)
    audio_format.setChannelCount(_CHANNEL_COUNT)
    audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    return audio_format


class PcmAudioCapture(QObject):
    """采集 16kHz/mono/Int16 裸 PCM 音频，通过信号增量交付。"""

    chunk_captured = Signal(bytes)
    capture_unavailable = Signal(str)
    max_duration_reached = Signal()

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        max_duration_s: int = 60,
        check_permission: Callable[[], Qt.PermissionStatus] | None = None,
        request_permission: Callable[[Callable[[Qt.PermissionStatus], None]], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._max_duration_ms = max_duration_s * 1000
        self._check_permission = check_permission or default_check_microphone_permission
        self._request_permission = request_permission or default_request_microphone_permission
        self._source: QAudioSource | None = None
        self._device: QIODevice | None = None

        self._max_duration_timer = QTimer(self)
        self._max_duration_timer.setSingleShot(True)
        self._max_duration_timer.timeout.connect(self._on_max_duration_reached)

    def start_capture(self) -> None:
        _, microphone_available = probe_capture_availability()
        if not microphone_available:
            self.capture_unavailable.emit("未检测到可用的麦克风")
            return

        status = self._check_permission()
        if status == Qt.PermissionStatus.Denied:
            self.capture_unavailable.emit("麦克风权限被拒绝，请在系统设置中允许后重试")
        elif status == Qt.PermissionStatus.Undetermined:
            self._request_permission(self._on_permission_result)
        else:
            self._start_source()

    def _on_permission_result(self, status: Qt.PermissionStatus) -> None:
        if status != Qt.PermissionStatus.Granted:
            self.capture_unavailable.emit("麦克风权限被拒绝，请在系统设置中允许后重试")
            return
        self._start_source()

    def _start_source(self) -> None:
        try:
            source = QAudioSource(QMediaDevices.defaultAudioInput(), _pcm_audio_format(), self)
            source.stateChanged.connect(self._on_state_changed)
            device = source.start()
            if device is None:
                raise RuntimeError("QAudioSource.start() 未返回可读设备")
            device.readyRead.connect(self._on_ready_read)
            self._source = source
            self._device = device
            self._max_duration_timer.start(self._max_duration_ms)
        except Exception:
            logger.exception("启动麦克风采集失败")
            self._cleanup()
            self.capture_unavailable.emit("启动麦克风采集失败")

    def _on_ready_read(self) -> None:
        device = self._device
        if device is None:
            return
        data = bytes(device.readAll().data())
        if data:
            self.chunk_captured.emit(data)

    def _on_state_changed(self, state: QAudio.State) -> None:
        if state != QAudio.State.StoppedState:
            return
        source = self._source
        if source is not None and source.error() != QtAudio.Error.NoError:
            logger.warning("麦克风采集异常：%s", source.error())
            self._max_duration_timer.stop()
            self._cleanup()
            self.capture_unavailable.emit("麦克风采集异常，请重试")

    def _on_max_duration_reached(self) -> None:
        self.stop_capture()
        self.max_duration_reached.emit()

    def stop_capture(self) -> None:
        self._max_duration_timer.stop()
        if self._source is not None:
            self._source.stop()
        self._cleanup()

    def _cleanup(self) -> None:
        self._source = None
        self._device = None
