"""更换声音向导：为已有角色重新绑定专属声音。

跟 ``character_clone_dialog.py`` 同款约定：``QWidget`` 而非 ``QDialog`` +
``setWindowFlags(Tool | FramelessWindowHint)`` 规避 QTBUG-83490；``QStackedLayout``
承载模式选择/录音克隆/手填三个页面。"重新录音克隆"直接复用 ``ReadingRecordingStepWidget``
+ ``VoiceCloneWorker`` + ``_VoiceCloneStatusCard``（跟克隆向导第 3/4 步同一套组件），
"手填已有声音"是纯本地表单，不发任何网络请求。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QStackedLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    ComboBox,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    StrongBodyLabel,
)

from miku_on_desk.brain.tts.voice_clone import VoiceCloneConfig
from miku_on_desk.config.settings import AppSettings, TTSProviderName, load_settings_with_vault
from miku_on_desk.face.character_voice import (
    PetVoiceConfig,
    delete_pet_voice_config,
    load_pet_voice_config,
    save_pet_voice_config,
)
from miku_on_desk.face.ui.character_clone_dialog import _DEFAULT_DESCRIPTION, _VoiceCloneStatusCard
from miku_on_desk.face.ui.reading_recording_step import ReadingRecordingStepWidget
from miku_on_desk.face.ui.theme import ERROR_COLOR, SPACING_MD, SPACING_SM
from miku_on_desk.face.voice_clone_worker import VoiceCloneWorker

if TYPE_CHECKING:
    from miku_on_desk.brain.secrets.vault import SecretVault


class VoiceChangeDialog(QWidget):
    """更换声音向导：模式选择 → （重新录音克隆 | 手填已有声音）→ 保存。"""

    voice_updated = Signal(Path)

    def __init__(
        self,
        pet_dir: Path,
        settings_path: Path,
        parent: QWidget | None = None,
        *,
        vault: SecretVault | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setWindowTitle("更换声音")
        self.resize(480, 420)
        self._pet_dir = pet_dir
        self._settings_path = settings_path
        self._vault = vault
        self._settings = self._load_settings()

        self._reading_recording_widget: ReadingRecordingStepWidget | None = None
        self._voice_worker: VoiceCloneWorker | None = None
        self._voice_card: _VoiceCloneStatusCard | None = None

        self._stack = QStackedLayout(self)
        self._mode_select_view = self._build_mode_select_view()
        self._stack.addWidget(self._mode_select_view)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self._voice_worker is None:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._reading_recording_widget is not None:
            self._reading_recording_widget.shutdown()
        super().closeEvent(event)

    def _load_settings(self) -> AppSettings:
        if self._vault is not None:
            return load_settings_with_vault(self._settings_path, self._vault)
        return AppSettings.load(self._settings_path)

    # ------------------------------------------------------------------
    # 模式选择页
    # ------------------------------------------------------------------

    def _build_mode_select_view(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setSpacing(SPACING_MD)

        self._mode_error_label = CaptionLabel("", container)
        self._mode_error_label.setStyleSheet(f"color: {ERROR_COLOR};")
        self._mode_error_label.hide()
        layout.addWidget(self._mode_error_label)

        title = StrongBodyLabel("选择更换方式", container)
        layout.addWidget(title)

        record_button = PrimaryPushButton("重新录音克隆", container)
        record_button.clicked.connect(self._show_record_mode)
        layout.addWidget(record_button)

        manual_button = PushButton("手填已有声音", container)
        manual_button.clicked.connect(self._show_manual_mode)
        layout.addWidget(manual_button)

        layout.addStretch(1)

        button_row = QHBoxLayout()
        button_row.setSpacing(SPACING_SM)
        restore_button = PushButton("恢复默认声音", container)
        restore_button.clicked.connect(self._on_restore_default_clicked)
        button_row.addWidget(restore_button)
        button_row.addStretch(1)
        close_button = PushButton("关闭", container)
        close_button.clicked.connect(self.close)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        return container

    def _on_restore_default_clicked(self) -> None:
        delete_pet_voice_config(self._pet_dir)
        self.voice_updated.emit(self._pet_dir)
        self.close()

    # ------------------------------------------------------------------
    # 模式 A：重新录音克隆
    # ------------------------------------------------------------------

    def _show_record_mode(self) -> None:
        if not self._settings.voice_cloning.elevenlabs_api_key:
            self._mode_error_label.setText(
                "未填写 ElevenLabs API Key，请先在设置面板的“语音”标签页填写后再试"
            )
            self._mode_error_label.show()
            return

        widget = ReadingRecordingStepWidget(self._settings.model_router, _DEFAULT_DESCRIPTION, self)
        widget.recorded.connect(self._on_recording_recorded)
        widget.skip_requested.connect(self._on_recording_skipped)
        self._reading_recording_widget = widget

        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.addWidget(widget)
        self._stack.addWidget(page)
        self._stack.setCurrentWidget(page)
        widget.start()

    def _on_recording_skipped(self) -> None:
        self._reading_recording_widget = None
        self._stack.setCurrentWidget(self._mode_select_view)

    def _on_recording_recorded(self, wav_bytes: bytes) -> None:
        self._reading_recording_widget = None
        self._show_cloning_page(wav_bytes)

    def _show_cloning_page(self, wav_bytes: bytes) -> None:
        card = _VoiceCloneStatusCard(self)
        self._voice_card = card

        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.addWidget(card)
        layout.addStretch(1)
        close_button = PushButton("关闭", page)
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        self._stack.addWidget(page)
        self._stack.setCurrentWidget(page)

        elevenlabs_api_key = self._settings.voice_cloning.elevenlabs_api_key
        assert elevenlabs_api_key is not None
        voice_config = VoiceCloneConfig(
            name=self._pet_dir.name,
            audio_bytes=wav_bytes,
            api_key=elevenlabs_api_key,
            base_url=self._settings.voice_cloning.elevenlabs_base_url,
        )
        card.start_cloning()
        worker = VoiceCloneWorker(voice_config, self)
        worker.finished_ok.connect(self._on_voice_done)
        worker.failed.connect(self._on_voice_failed)
        self._voice_worker = worker
        worker.start()

    def _on_voice_done(self, voice_id: str) -> None:
        self._voice_worker = None
        save_pet_voice_config(
            self._pet_dir, PetVoiceConfig(provider=TTSProviderName.ELEVENLABS, voice=voice_id)
        )
        if self._voice_card is not None:
            self._voice_card.show_success()
        self.voice_updated.emit(self._pet_dir)

    def _on_voice_failed(self, message: str) -> None:
        self._voice_worker = None
        if self._voice_card is not None:
            self._voice_card.show_failure(message)

    # ------------------------------------------------------------------
    # 模式 B：手填已有声音
    # ------------------------------------------------------------------

    def _show_manual_mode(self) -> None:
        page = self._build_manual_page()
        self._stack.addWidget(page)
        self._stack.setCurrentWidget(page)

    def _build_manual_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)

        self._manual_error_label = CaptionLabel("", page)
        self._manual_error_label.setStyleSheet(f"color: {ERROR_COLOR};")
        self._manual_error_label.hide()
        layout.addWidget(self._manual_error_label)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._manual_provider_combo = ComboBox(page)
        self._manual_provider_combo.addItems([provider.value for provider in TTSProviderName])
        form.addRow("Provider", self._manual_provider_combo)

        self._manual_voice_edit = LineEdit(page)
        self._manual_voice_edit.setPlaceholderText("音库名 / voice_id")
        form.addRow("声音", self._manual_voice_edit)

        self._manual_model_edit = LineEdit(page)
        self._manual_model_edit.setPlaceholderText("留空则使用默认值")
        form.addRow("模型（可选）", self._manual_model_edit)

        existing = load_pet_voice_config(self._pet_dir)
        if existing is not None:
            self._manual_provider_combo.setCurrentText(existing.provider.value)
            self._manual_voice_edit.setText(existing.voice)
            self._manual_model_edit.setText(existing.model)

        layout.addLayout(form)
        layout.addStretch(1)

        button_row = QHBoxLayout()
        back_button = PushButton("返回", page)
        back_button.clicked.connect(lambda: self._stack.setCurrentWidget(self._mode_select_view))
        button_row.addWidget(back_button)
        button_row.addStretch(1)
        save_button = PrimaryPushButton("保存", page)
        save_button.clicked.connect(self._on_manual_save_clicked)
        button_row.addWidget(save_button)
        layout.addLayout(button_row)

        return page

    def _on_manual_save_clicked(self) -> None:
        voice = self._manual_voice_edit.text().strip()
        if not voice:
            self._manual_error_label.setText("请填写声音 / voice_id")
            self._manual_error_label.show()
            return
        provider = TTSProviderName(self._manual_provider_combo.currentText())
        model = self._manual_model_edit.text().strip()
        config = (
            PetVoiceConfig(provider=provider, voice=voice, model=model)
            if model
            else PetVoiceConfig(provider=provider, voice=voice)
        )
        save_pet_voice_config(self._pet_dir, config)
        self.voice_updated.emit(self._pet_dir)
        self.close()
