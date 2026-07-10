"""聊天输入弹窗：右键圆环菜单"对miku说"或托盘菜单触发的无边框气泡式输入框。

用 ``Qt.WindowType.Tool`` 而非 ``Qt.WindowType.Popup``：后者在 macOS/Windows 上有一个
长期未解决的上游 Qt bug（QTBUG-83490），Popup 窗口在系统层面成不了正常的"key window"，
导致输入法合成拿不到正确的焦点上下文——中文输入法只能把拼音字母原样插入，跳过候选词/
上屏。改用 ``Tool`` 后失去了 Popup 自带的"点外面自动关闭"，靠 ``changeEvent`` 监听
失焦手动补上。

语音输入（``voice_capture``/``stt_worker`` 均非 ``None`` 时才启用）走点击切换 + 实时流式
转写：转写结果只写入输入框供用户确认/编辑，从不自动发送——ElevenLabs 中文识别准确率官方
文档标注为"中等"档，静默自动发送有把误转写当用户原话发出去的风险。用户手动打字
（``QLineEdit.textEdited``，只在真正键盘/粘贴输入时触发，程序调用 ``setText`` 不会触发）
会立刻打断录音，避免"转写自动填入"和"用户正在编辑"互相打架。

点击麦克风开始录音时会 emit ``barge_in_requested``——这是"点击麦克风即打断"的信号来源：
本弹窗不直接持有取消生成/停止播放所需的对象，只负责在用户明确要开始说话的瞬间往上抛信号，
交给持有这些对象的 ``OverlayWindow`` 处理。
"""

from __future__ import annotations

from enum import StrEnum

from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QWidget
from qfluentwidgets import FluentIcon, ToggleToolButton

from miku_on_desk.face.stt_worker import SttWorker
from miku_on_desk.face.ui.audio_capture import PcmAudioCapture
from miku_on_desk.face.ui.theme import RADIUS_XL, TEAL_DARK

_WIDTH = 320
_HEIGHT = 44
_MIC_BUTTON_SIZE = 32
_FINALIZE_TIMEOUT_MS = 5000
_ERROR_DISPLAY_MS = 2500
_DEFAULT_PLACEHOLDER = "对 Miku 说点什么…"

_INPUT_STYLE = f"""
QLineEdit {{
    background-color: rgba(143, 218, 198, 200);
    border: 2px solid {TEAL_DARK};
    border-radius: {RADIUS_XL}px;
    padding: 6px 14px;
    color: #1a1a1a;
    font-size: 14px;
}}
"""


class _MicState(StrEnum):
    IDLE = "idle"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    ERROR = "error"


class ChatPopup(QWidget):
    """默认隐藏；调用 ``popup_at`` 定位、显示并聚焦输入框。"""

    text_submitted = Signal(str)
    barge_in_requested = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        voice_capture: PcmAudioCapture | None = None,
        stt_worker: SttWorker | None = None,
    ) -> None:
        super().__init__(parent)
        self._voice_capture = voice_capture
        self._stt_worker = stt_worker
        has_voice_input = voice_capture is not None and stt_worker is not None

        self._mic_state = _MicState.IDLE
        self._active_session_id: int | None = None
        self._committed_prefix = ""

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        width = _WIDTH + _MIC_BUTTON_SIZE + 8 if has_voice_input else _WIDTH
        self.resize(width, _HEIGHT)

        self._input = QLineEdit(self)
        self._input.setPlaceholderText(_DEFAULT_PLACEHOLDER)
        self._input.setStyleSheet(_INPUT_STYLE)
        self._input.returnPressed.connect(self._on_submit)
        self._input.textEdited.connect(self._on_text_edited)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._input)

        self._mic_button: ToggleToolButton | None = None
        if has_voice_input:
            mic_button = ToggleToolButton(self)
            mic_button.setIcon(FluentIcon.MICROPHONE)
            mic_button.setFixedSize(_MIC_BUTTON_SIZE, _MIC_BUTTON_SIZE)
            mic_button.setToolTip("点击开始/结束语音输入")
            mic_button.clicked.connect(self._on_mic_button_clicked)
            layout.addWidget(mic_button)
            self._mic_button = mic_button

            assert self._voice_capture is not None
            assert self._stt_worker is not None
            self._voice_capture.chunk_captured.connect(self._on_chunk_captured)
            self._voice_capture.capture_unavailable.connect(self._on_capture_unavailable)
            self._voice_capture.max_duration_reached.connect(self._on_max_duration_reached)
            self._stt_worker.partial_transcript.connect(self._on_partial_transcript)
            self._stt_worker.committed_transcript.connect(self._on_committed_transcript)
            self._stt_worker.session_error.connect(self._on_session_error)
            self._stt_worker.session_closed.connect(self._on_session_closed)

    def popup_at(self, global_pos: QPoint) -> None:
        self.move(global_pos.x(), global_pos.y())
        self._input.clear()
        self.show()
        self.activateWindow()
        QTimer.singleShot(0, self._input.setFocus)

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow():
            self.close()
            return
        super().changeEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._interrupt_recording()
        super().closeEvent(event)

    def _on_submit(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self.text_submitted.emit(text)
        self.close()

    # -- 麦克风状态机 --------------------------------------------------------

    def _set_mic_state(self, state: _MicState) -> None:
        self._mic_state = state
        button = self._mic_button
        if button is None:
            return
        if state == _MicState.IDLE:
            button.setChecked(False)
            button.setEnabled(True)
        elif state == _MicState.RECORDING:
            button.setChecked(True)
            button.setEnabled(True)
        else:  # FINALIZING / ERROR
            button.setChecked(False)
            button.setEnabled(False)

    def _on_mic_button_clicked(self, checked: bool) -> None:
        if checked:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        if self._voice_capture is None or self._stt_worker is None:
            return
        self.barge_in_requested.emit()
        self._committed_prefix = self._input.text()
        self._active_session_id = self._stt_worker.begin_session()
        self._voice_capture.start_capture()
        self._set_mic_state(_MicState.RECORDING)

    def _stop_recording(self) -> None:
        if self._voice_capture is not None:
            self._voice_capture.stop_capture()
        self._begin_finalizing()

    def _begin_finalizing(self) -> None:
        if self._stt_worker is not None and self._active_session_id is not None:
            self._stt_worker.end_session(self._active_session_id)
        self._set_mic_state(_MicState.FINALIZING)
        QTimer.singleShot(_FINALIZE_TIMEOUT_MS, self._on_finalize_timeout)

    def _interrupt_recording(self) -> None:
        if self._mic_state not in (_MicState.RECORDING, _MicState.FINALIZING):
            return
        if self._voice_capture is not None:
            self._voice_capture.stop_capture()
        if self._stt_worker is not None and self._active_session_id is not None:
            self._stt_worker.end_session(self._active_session_id)
        self._active_session_id = None
        self._set_mic_state(_MicState.IDLE)

    def _on_text_edited(self, _text: str) -> None:
        self._interrupt_recording()

    def _on_chunk_captured(self, chunk: bytes) -> None:
        if self._active_session_id is not None and self._stt_worker is not None:
            self._stt_worker.push_chunk(self._active_session_id, chunk)

    def _on_partial_transcript(self, session_id: int, text: str) -> None:
        if session_id != self._active_session_id:
            return
        self._input.setText(self._committed_prefix + text)

    def _on_committed_transcript(self, session_id: int, text: str) -> None:
        if session_id != self._active_session_id:
            return
        self._committed_prefix += text
        self._input.setText(self._committed_prefix)

    def _on_session_error(self, session_id: int, message: str) -> None:
        if session_id != self._active_session_id:
            return
        self._show_error(message)

    def _on_session_closed(self, session_id: int) -> None:
        if session_id != self._active_session_id:
            return
        self._active_session_id = None
        if self._mic_state == _MicState.FINALIZING:
            self._set_mic_state(_MicState.IDLE)

    def _on_capture_unavailable(self, message: str) -> None:
        if self._stt_worker is not None and self._active_session_id is not None:
            self._stt_worker.end_session(self._active_session_id)
        self._show_error(message)

    def _on_max_duration_reached(self) -> None:
        if self._mic_state != _MicState.RECORDING:
            return
        self._begin_finalizing()

    def _on_finalize_timeout(self) -> None:
        if self._mic_state == _MicState.FINALIZING:
            self._active_session_id = None
            self._set_mic_state(_MicState.IDLE)

    def _show_error(self, message: str) -> None:
        self._active_session_id = None
        if self._voice_capture is not None:
            self._voice_capture.stop_capture()
        self._input.setPlaceholderText(message)
        self._set_mic_state(_MicState.ERROR)
        QTimer.singleShot(_ERROR_DISPLAY_MS, self._on_error_display_timeout)

    def _on_error_display_timeout(self) -> None:
        if self._mic_state == _MicState.ERROR:
            self._input.setPlaceholderText(_DEFAULT_PLACEHOLDER)
            self._set_mic_state(_MicState.IDLE)
