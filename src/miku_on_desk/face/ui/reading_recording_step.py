"""朗读文本生成 + 30 秒录音的可复用步骤组件。

克隆向导第 3 步与"更换声音"对话框的"重新录音克隆"模式共用这一组件：``start()``
触发朗读文本生成，文本淡入后才启动 ``AudioRecorderWidget``（避免 LLM 延迟占用 30 秒
录音窗口）。录音完成/麦克风不可用两种终态分别对应 ``recorded``/``skip_requested``
信号，调用方据此决定是否继续声音克隆。

跟 ``AudioRecorderWidget``/``CameraCaptureWidget`` 同款约定：构造后处于静止状态，调用方
显式调 ``start()`` 才开始工作——真正触碰线程/硬件的私有方法（``_start_script_generation``）
可以在测试里被整个替换掉，不需要真的起线程。
"""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal
from PySide6.QtWidgets import QGraphicsOpacityEffect, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, PlainTextEdit, PrimaryPushButton, PushButton

from miku_on_desk.config.settings import ModelRouterConfig
from miku_on_desk.face.reading_script_worker import ReadingScriptWorker
from miku_on_desk.face.ui.capture_widgets import AudioRecorderWidget
from miku_on_desk.face.ui.theme import (
    ERROR_COLOR,
    PLACEHOLDER_BG,
    RADIUS_MD,
    SPACING_MD,
    SPACING_SM,
    TEAL_DARK,
    border_qss,
)

_BREATH_DURATION_MS = 1200
_RESULT_FADE_MS = 200
_SCRIPT_HEIGHT = 120


class ReadingRecordingStepWidget(QWidget):
    """朗读文本生成 + 30 秒录音；``recorded`` 携带最终 WAV bytes，``skip_requested`` 表示跳过。"""

    recorded = Signal(bytes)
    skip_requested = Signal()

    def __init__(
        self,
        model_router_config: ModelRouterConfig,
        description: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model_router_config = model_router_config
        self._description = description
        self._worker: ReadingScriptWorker | None = None
        self._recorded_bytes: bytes | None = None
        self._recording_available = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING_MD)

        self._script_edit = PlainTextEdit(self)
        self._script_edit.setReadOnly(True)
        self._script_edit.setFixedHeight(_SCRIPT_HEIGHT)
        self._script_edit.setStyleSheet(
            f"background-color: {PLACEHOLDER_BG}; {border_qss(TEAL_DARK, radius=RADIUS_MD)}"
        )
        layout.addWidget(self._script_edit)

        self._script_opacity_effect = QGraphicsOpacityEffect(self._script_edit)
        self._script_opacity_effect.setOpacity(1.0)
        self._script_edit.setGraphicsEffect(self._script_opacity_effect)

        self._breath_anim = QPropertyAnimation(self._script_opacity_effect, b"opacity", self)
        self._breath_anim.setDuration(_BREATH_DURATION_MS)
        self._breath_anim.setKeyValueAt(0.0, 0.35)
        self._breath_anim.setKeyValueAt(0.5, 1.0)
        self._breath_anim.setKeyValueAt(1.0, 0.35)
        self._breath_anim.setLoopCount(-1)

        self._recorder = AudioRecorderWidget(self)
        self._recorder.recording_finished.connect(self._on_recording_finished)
        self._recorder.recording_unavailable.connect(self._on_recording_unavailable)
        self._recorder.seconds_remaining_changed.connect(self._on_seconds_remaining_changed)
        layout.addWidget(self._recorder)

        self._countdown_label = CaptionLabel("", self)
        self._countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._countdown_label)

        self._error_label = CaptionLabel("", self)
        self._error_label.setStyleSheet(f"color: {ERROR_COLOR};")
        self._error_label.hide()
        layout.addWidget(self._error_label)

        button_row = QHBoxLayout()
        button_row.setSpacing(SPACING_SM)
        self._retry_button = PushButton("重新生成文本并重录", self)
        self._retry_button.clicked.connect(self._on_retry_clicked)
        button_row.addWidget(self._retry_button)
        button_row.addStretch(1)
        self._next_button = PrimaryPushButton("下一步", self)
        self._next_button.setEnabled(False)
        self._next_button.clicked.connect(self._on_next_clicked)
        button_row.addWidget(self._next_button)
        layout.addLayout(button_row)

    def start(self) -> None:
        self._start_script_generation()

    def shutdown(self) -> None:
        """外部（向导关闭/切换步骤）调用：取消朗读文本生成、停止录音，避免残留线程/硬件占用。"""
        if self._worker is not None:
            self._worker.request_cancel()
            self._worker.wait(3000)
            self._worker = None
        self._recorder.stop()

    def _start_script_generation(self) -> None:
        self._recording_available = True
        self._recorded_bytes = None
        self._error_label.hide()
        self._next_button.setEnabled(False)
        self._next_button.setText("下一步")
        self._countdown_label.setText("")
        self._script_opacity_effect.setOpacity(1.0)
        self._script_edit.setPlainText("正在生成朗读文本…")
        self._breath_anim.start()

        worker = ReadingScriptWorker(self._description, self._model_router_config, self)
        worker.finished_ok.connect(self._on_script_ready)
        worker.failed.connect(self._on_script_failed)
        self._worker = worker
        worker.start()

    def _on_script_ready(self, text: str) -> None:
        self._worker = None
        self._breath_anim.stop()
        self._script_opacity_effect.setOpacity(0.0)
        self._script_edit.setPlainText(text)

        fade = QPropertyAnimation(self._script_opacity_effect, b"opacity", self)
        fade.setDuration(_RESULT_FADE_MS)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        fade.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        self._fade_anim = fade

        self._recorder.start()

    def _on_script_failed(self, message: str) -> None:
        self._worker = None
        self._breath_anim.stop()
        self._script_opacity_effect.setOpacity(1.0)
        self._script_edit.setPlainText("")
        self._error_label.setText(f"朗读文本生成失败：{message}")
        self._error_label.show()

    def _on_retry_clicked(self) -> None:
        self._recorder.stop()
        self._start_script_generation()

    def _on_recording_finished(self, wav_bytes: bytes) -> None:
        self._recorded_bytes = wav_bytes
        self._countdown_label.setText("录音完成")
        self._next_button.setEnabled(True)

    def _on_recording_unavailable(self, reason: str) -> None:
        self._recording_available = False
        self._recorded_bytes = None
        self._countdown_label.setText("")
        self._error_label.setText(reason)
        self._error_label.show()
        self._next_button.setText("跳过声音克隆，仅生成外观")
        self._next_button.setEnabled(True)

    def _on_seconds_remaining_changed(self, seconds: int) -> None:
        self._countdown_label.setText(f"录音中… 剩余 {seconds} 秒")

    def _on_next_clicked(self) -> None:
        if not self._recording_available:
            self.skip_requested.emit()
            return
        if self._recorded_bytes is not None:
            self.recorded.emit(self._recorded_bytes)
