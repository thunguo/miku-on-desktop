"""克隆角色向导：表单 → 拍照 → 朗读录音 → 并行生成（像素外观 + 克隆声音）→ 完成。

跟 ``character_creation_dialog.py`` 同款约定：``QWidget`` 而非 ``QDialog`` +
``setWindowFlags(Tool | FramelessWindowHint)`` 规避 QTBUG-83490；``QStackedLayout``
承载五个页面，全程同一个窗口，不额外弹第二个窗口。像素生成那一半直接复用
``_GenerationProgressView``/``CharacterGenerationWorker``（不做任何改动），声音克隆
那一半新增一个视觉家族一致的 ``_VoiceCloneStatusCard``。

两个 worker 各自独立完成、各自更新自己的状态展示；只有当"像素成功 且（声音已有结果
或已确定跳过）"时才进入完成页——像素失败或被取消时立即让声音 worker 一起取消并直接
判整个向导失败，不等声音结果，因为像素失败已经决定了向导的最终结局。
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from PIL import Image
from PySide6.QtCore import QPropertyAnimation, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QCloseEvent, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QFormLayout,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CaptionLabel,
    ComboBox,
    LineEdit,
    PlainTextEdit,
    PrimaryPushButton,
    PushButton,
    StrongBodyLabel,
)

from miku_on_desk.brain.tts.voice_clone import VoiceCloneConfig
from miku_on_desk.character_generation import GenerationConfig
from miku_on_desk.config.settings import (
    AppSettings,
    TTSProviderName,
    load_settings_with_vault,
    save_settings_with_vault,
)
from miku_on_desk.face.character_generation_worker import CharacterGenerationWorker
from miku_on_desk.face.character_voice import PetVoiceConfig, save_pet_voice_config
from miku_on_desk.face.ui.capture_widgets import CameraCaptureWidget
from miku_on_desk.face.ui.character_creation_dialog import (
    _BREATH_DURATION_MS,
    _GLOW_DURATION_MS,
    _GLOW_FADE_MS,
    _MODEL_CHOICES,
    _NAME_PATTERN,
    _GenerationProgressView,
    _pil_to_pixmap,
)
from miku_on_desk.face.ui.reading_recording_step import ReadingRecordingStepWidget
from miku_on_desk.face.ui.theme import (
    ERROR_COLOR,
    PLACEHOLDER_BG,
    RADIUS_LG,
    RADIUS_MD,
    SPACING_XXS,
    TEAL_DARK,
    TEAL_MAIN,
    border_qss,
    qcolor,
)
from miku_on_desk.face.voice_clone_worker import VoiceCloneWorker

if TYPE_CHECKING:
    from miku_on_desk.brain.secrets.vault import SecretVault

_PHOTO_PREVIEW_SIZE = 240
_COMPLETION_THUMBNAIL_SIZE = 160
_DEFAULT_DESCRIPTION = "一个可爱的虚拟角色伙伴"


class _VoiceCloneStatusCard(QWidget):
    """声音克隆状态小卡片：等待中 → 克隆中（呼吸）→ 成功（高光褪色）/失败/跳过。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_XXS, SPACING_XXS, SPACING_XXS, SPACING_XXS)

        self._idle_style = f"background-color: {PLACEHOLDER_BG}; border-radius: {RADIUS_MD}px;"
        self._done_style = (
            f"background-color: {PLACEHOLDER_BG}; {border_qss(TEAL_DARK, radius=RADIUS_MD)}"
        )
        self._failed_style = (
            f"background-color: {PLACEHOLDER_BG}; {border_qss(ERROR_COLOR, radius=RADIUS_MD)}"
        )

        self._status_label = StrongBodyLabel("等待声音克隆…", self)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        self._status_label.setMinimumHeight(64)
        self._status_label.setStyleSheet(self._idle_style)
        layout.addWidget(self._status_label)

        caption = CaptionLabel("声音克隆", self)
        caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(caption)

        self._opacity_effect = QGraphicsOpacityEffect(self._status_label)
        self._opacity_effect.setOpacity(1.0)
        self._status_label.setGraphicsEffect(self._opacity_effect)

        self._breath_anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._breath_anim.setDuration(_BREATH_DURATION_MS)
        self._breath_anim.setKeyValueAt(0.0, 0.35)
        self._breath_anim.setKeyValueAt(0.5, 1.0)
        self._breath_anim.setKeyValueAt(1.0, 0.35)
        self._breath_anim.setLoopCount(-1)

    def start_cloning(self) -> None:
        self._status_label.setText("声音克隆中…")
        self._status_label.setStyleSheet(self._idle_style)
        self._breath_anim.start()

    def show_skipped(self, message: str = "已跳过声音克隆") -> None:
        self._breath_anim.stop()
        self._opacity_effect.setOpacity(1.0)
        self._status_label.setText(message)
        self._status_label.setStyleSheet(self._idle_style)

    def show_failure(self, message: str) -> None:
        self._breath_anim.stop()
        self._opacity_effect.setOpacity(1.0)
        self._status_label.setText(f"声音克隆失败：{message}")
        self._status_label.setStyleSheet(self._failed_style)

    def show_success(self) -> None:
        self._breath_anim.stop()
        self._opacity_effect.setOpacity(1.0)
        self._status_label.setText("声音克隆完成")
        self._status_label.setStyleSheet(self._done_style)
        self.setStyleSheet(f"_VoiceCloneStatusCard {{ {border_qss(TEAL_MAIN)} }}")
        QTimer.singleShot(_GLOW_DURATION_MS, self._start_glow_fade)

    def _start_glow_fade(self) -> None:
        fade = QVariantAnimation(self)
        fade.setDuration(_GLOW_FADE_MS)
        fade.setStartValue(255)
        fade.setEndValue(0)
        fade.valueChanged.connect(self._apply_glow_alpha)
        fade.start(QVariantAnimation.DeletionPolicy.DeleteWhenStopped)
        self._glow_fade_anim = fade

    def _apply_glow_alpha(self, alpha: int) -> None:
        color = qcolor(TEAL_MAIN, alpha=alpha)
        self.setStyleSheet(
            f"_VoiceCloneStatusCard {{ border: 2px solid "
            f"rgba({color.red()}, {color.green()}, {color.blue()}, {alpha}); "
            f"border-radius: {RADIUS_LG}px; }}"
        )


class CharacterCloneDialog(QWidget):
    """克隆角色向导：表单 → 拍照 → 朗读录音 → 并行生成 → 完成，全程同一个窗口。"""

    character_created = Signal(Path)

    def __init__(
        self,
        assets_pets_dir: Path,
        settings_path: Path,
        parent: QWidget | None = None,
        *,
        vault: SecretVault | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setWindowTitle("克隆角色")
        self.resize(560, 480)
        self._assets_pets_dir = assets_pets_dir
        self._settings_path = settings_path
        self._vault = vault
        self._settings: AppSettings | None = None

        self._generation_config: GenerationConfig | None = None
        self._elevenlabs_api_key: str | None = None
        self._elevenlabs_base_url: str | None = None
        self._captured_photo_bytes: bytes | None = None
        self._recorded_wav_bytes: bytes | None = None
        self._voice_skipped = False

        self._camera_widget: CameraCaptureWidget | None = None
        self._reading_recording_widget: ReadingRecordingStepWidget | None = None
        self._progress_view: _GenerationProgressView | None = None
        self._voice_card: _VoiceCloneStatusCard | None = None
        self._pixel_worker: CharacterGenerationWorker | None = None
        self._voice_worker: VoiceCloneWorker | None = None
        self._pixel_result: tuple[Image.Image, list[str]] | None = None
        self._voice_result: tuple[Literal["ok", "failed"], str] | None = None
        self._output_dir: Path | None = None

        self._stack = QStackedLayout(self)
        self._form_view = self._build_form_view()
        self._stack.addWidget(self._form_view)
        self._prefill_from_settings()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if (
            event.key() == Qt.Key.Key_Escape
            and self._pixel_worker is None
            and self._voice_worker is None
        ):
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._camera_widget is not None:
            self._camera_widget.stop()
        if self._reading_recording_widget is not None:
            self._reading_recording_widget.shutdown()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # 第 1 页：表单
    # ------------------------------------------------------------------

    def _build_form_view(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)

        self._error_label = CaptionLabel("", container)
        self._error_label.setStyleSheet(f"color: {ERROR_COLOR};")
        self._error_label.hide()
        layout.addWidget(self._error_label)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._name_edit = LineEdit(container)
        self._name_edit.setPlaceholderText("仅限字母/数字/下划线/短横线")
        form.addRow("角色名称", self._name_edit)

        self._description_edit = PlainTextEdit(container)
        self._description_edit.setPlaceholderText(
            "用文字描述这个角色的性格、气质…（可选，留空则使用默认描述）"
        )
        form.addRow("角色描述", self._description_edit)

        self._base_url_edit = LineEdit(container)
        self._base_url_edit.setPlaceholderText("默认官方地址")
        form.addRow("图像生成 API Base URL", self._base_url_edit)

        self._api_key_edit = LineEdit(container)
        self._api_key_edit.setEchoMode(LineEdit.EchoMode.Password)
        form.addRow("图像生成 API Key", self._api_key_edit)

        self._model_combo = ComboBox(container)
        self._model_combo.addItems(list(_MODEL_CHOICES))
        form.addRow("模型", self._model_combo)

        self._elevenlabs_base_url_edit = LineEdit(container)
        self._elevenlabs_base_url_edit.setPlaceholderText("默认官方地址")
        form.addRow("ElevenLabs API Base URL", self._elevenlabs_base_url_edit)

        self._elevenlabs_api_key_edit = LineEdit(container)
        self._elevenlabs_api_key_edit.setEchoMode(LineEdit.EchoMode.Password)
        self._elevenlabs_api_key_edit.setPlaceholderText("留空则跳过声音克隆")
        form.addRow("ElevenLabs API Key", self._elevenlabs_api_key_edit)

        layout.addLayout(form)
        layout.addStretch(1)

        button_row = QHBoxLayout()
        close_button = PushButton("关闭", container)
        close_button.clicked.connect(self.close)
        button_row.addWidget(close_button)
        next_button = PrimaryPushButton("下一步：拍照", container)
        next_button.clicked.connect(self._on_form_next_clicked)
        button_row.addWidget(next_button)
        layout.addLayout(button_row)

        return container

    def _prefill_from_settings(self) -> None:
        if self._vault is not None:
            settings = load_settings_with_vault(self._settings_path, self._vault)
        else:
            settings = AppSettings.load(self._settings_path)
        self._settings = settings

        image_generation = settings.image_generation
        self._base_url_edit.setText(image_generation.base_url or "")
        self._api_key_edit.setText(image_generation.api_key or "")
        if image_generation.model in _MODEL_CHOICES:
            self._model_combo.setCurrentText(image_generation.model)

        voice_cloning = settings.voice_cloning
        self._elevenlabs_base_url_edit.setText(voice_cloning.elevenlabs_base_url or "")
        self._elevenlabs_api_key_edit.setText(voice_cloning.elevenlabs_api_key or "")

    def _show_error(self, message: str) -> None:
        self._error_label.setText(message)
        self._error_label.show()

    def _validate_form(self) -> tuple[GenerationConfig, str | None, str | None] | None:
        name = self._name_edit.text().strip()
        if not _NAME_PATTERN.match(name):
            self._show_error("角色名称只能包含字母、数字、下划线、短横线，且不能为空")
            return None
        output_dir = self._assets_pets_dir / name
        if output_dir.exists():
            self._show_error(f"角色名称 “{name}” 已存在，请换一个")
            return None

        api_key = self._api_key_edit.text().strip()
        if not api_key:
            self._show_error("请填写图像生成 API Key")
            return None

        description = self._description_edit.toPlainText().strip() or _DEFAULT_DESCRIPTION

        self._error_label.hide()
        config = GenerationConfig(
            pet_name=name,
            description=description,
            output_dir=output_dir,
            model=self._model_combo.currentText(),
            api_key=api_key,
            base_url=self._base_url_edit.text().strip() or None,
        )
        elevenlabs_api_key = self._elevenlabs_api_key_edit.text().strip() or None
        elevenlabs_base_url = self._elevenlabs_base_url_edit.text().strip() or None
        return config, elevenlabs_api_key, elevenlabs_base_url

    def _on_form_next_clicked(self) -> None:
        validated = self._validate_form()
        if validated is None:
            return
        config, elevenlabs_api_key, elevenlabs_base_url = validated
        self._generation_config = config
        self._elevenlabs_api_key = elevenlabs_api_key
        self._elevenlabs_base_url = elevenlabs_base_url

        settings = self._settings if self._settings is not None else AppSettings.load(
            self._settings_path
        )
        settings.image_generation.api_key = config.api_key
        settings.image_generation.base_url = config.base_url
        settings.image_generation.model = config.model
        settings.voice_cloning.elevenlabs_api_key = elevenlabs_api_key
        settings.voice_cloning.elevenlabs_base_url = elevenlabs_base_url
        if self._vault is not None:
            save_settings_with_vault(settings, self._settings_path, self._vault)
        else:
            settings.save(self._settings_path)
        self._settings = settings

        self._show_photo_page()

    # ------------------------------------------------------------------
    # 第 2 页：拍照
    # ------------------------------------------------------------------

    def _build_photo_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)

        self._photo_error_label = CaptionLabel("", page)
        self._photo_error_label.setStyleSheet(f"color: {ERROR_COLOR};")
        self._photo_error_label.hide()
        layout.addWidget(self._photo_error_label)

        camera_widget = CameraCaptureWidget(page)
        camera_widget.photo_captured.connect(self._on_photo_captured)
        camera_widget.capture_unavailable.connect(self._on_photo_capture_unavailable)
        self._camera_widget = camera_widget
        layout.addWidget(camera_widget)

        self._photo_preview_label = QLabel(page)
        self._photo_preview_label.setFixedSize(_PHOTO_PREVIEW_SIZE, _PHOTO_PREVIEW_SIZE)
        self._photo_preview_label.setStyleSheet(
            f"background-color: {PLACEHOLDER_BG}; {border_qss(TEAL_DARK, radius=RADIUS_MD)}"
        )
        self._photo_preview_label.hide()
        layout.addWidget(self._photo_preview_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        button_row = QHBoxLayout()
        self._capture_button = PushButton("拍照", page)
        self._capture_button.clicked.connect(self._on_capture_clicked)
        button_row.addWidget(self._capture_button)
        button_row.addStretch(1)
        self._photo_next_button = PrimaryPushButton("下一步", page)
        self._photo_next_button.setEnabled(False)
        self._photo_next_button.clicked.connect(self._on_photo_next_clicked)
        button_row.addWidget(self._photo_next_button)
        layout.addLayout(button_row)

        return page

    def _show_photo_page(self) -> None:
        page = self._build_photo_page()
        self._stack.addWidget(page)
        self._stack.setCurrentWidget(page)
        if self._camera_widget is not None:
            self._camera_widget.start()

    def _on_capture_clicked(self) -> None:
        if self._camera_widget is not None:
            self._camera_widget.capture_photo()

    def _on_photo_captured(self, png_bytes: bytes) -> None:
        self._captured_photo_bytes = png_bytes
        pixmap = QPixmap()
        pixmap.loadFromData(png_bytes)
        self._photo_preview_label.setPixmap(
            pixmap.scaled(
                _PHOTO_PREVIEW_SIZE,
                _PHOTO_PREVIEW_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._photo_preview_label.show()
        self._photo_next_button.setEnabled(True)

    def _on_photo_capture_unavailable(self, message: str) -> None:
        self._photo_error_label.setText(message)
        self._photo_error_label.show()
        self._capture_button.setEnabled(False)
        self._photo_next_button.setText("跳过拍照，仅用文字描述生成")
        self._photo_next_button.setEnabled(True)

    def _on_photo_next_clicked(self) -> None:
        if self._camera_widget is not None:
            self._camera_widget.stop()
            self._camera_widget = None
        self._show_reading_recording_page()

    # ------------------------------------------------------------------
    # 第 3 页：朗读 + 录音
    # ------------------------------------------------------------------

    def _show_reading_recording_page(self) -> None:
        config = self._generation_config
        settings = self._settings
        assert config is not None
        assert settings is not None

        widget = ReadingRecordingStepWidget(settings.model_router, config.description, self)
        widget.recorded.connect(self._on_recording_recorded)
        widget.skip_requested.connect(self._on_recording_skipped)
        self._reading_recording_widget = widget

        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.addWidget(widget)
        self._stack.addWidget(page)
        self._stack.setCurrentWidget(page)
        widget.start()

    def _on_recording_recorded(self, wav_bytes: bytes) -> None:
        self._recorded_wav_bytes = wav_bytes
        self._reading_recording_widget = None
        self._show_generation_page()

    def _on_recording_skipped(self) -> None:
        self._recorded_wav_bytes = None
        self._reading_recording_widget = None
        self._show_generation_page()

    # ------------------------------------------------------------------
    # 第 4 页：并行生成
    # ------------------------------------------------------------------

    def _config_with_reference_photo(
        self, config: GenerationConfig, png_bytes: bytes
    ) -> GenerationConfig:
        fd, temp_path_str = tempfile.mkstemp(suffix=".png", prefix="miku-clone-selfie-")
        with os.fdopen(fd, "wb") as fh:
            fh.write(png_bytes)
        return replace(
            config, reference_image_path=Path(temp_path_str), reference_image_kind="selfie"
        )

    def _show_generation_page(self) -> None:
        config = self._generation_config
        assert config is not None
        if self._captured_photo_bytes is not None:
            config = self._config_with_reference_photo(config, self._captured_photo_bytes)
            self._generation_config = config
        self._output_dir = config.output_dir

        self._pixel_result = None
        self._voice_result = None
        self._voice_skipped = self._recorded_wav_bytes is None or not self._elevenlabs_api_key

        progress_view = _GenerationProgressView(config.frame_width, config.frame_height, self)
        progress_view.cancel_requested.connect(self._on_cancel_generation)
        self._progress_view = progress_view

        voice_card = _VoiceCloneStatusCard(self)
        self._voice_card = voice_card

        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.addWidget(progress_view)
        layout.addWidget(voice_card)
        self._stack.addWidget(page)
        self._stack.setCurrentWidget(page)

        pixel_worker = CharacterGenerationWorker(config, self)
        pixel_worker.progress.connect(progress_view.on_progress)
        pixel_worker.finished_ok.connect(self._on_pixel_done)
        pixel_worker.failed.connect(self._on_pixel_failed)
        pixel_worker.cancelled.connect(self._on_pixel_cancelled)
        self._pixel_worker = pixel_worker
        pixel_worker.start()

        if self._voice_skipped:
            if self._recorded_wav_bytes is None:
                voice_card.show_skipped("未录制声音，已跳过声音克隆")
            else:
                voice_card.show_skipped("未填写 ElevenLabs API Key，已跳过声音克隆")
            return

        assert self._recorded_wav_bytes is not None
        assert self._elevenlabs_api_key is not None
        voice_config = VoiceCloneConfig(
            name=config.pet_name,
            audio_bytes=self._recorded_wav_bytes,
            api_key=self._elevenlabs_api_key,
            base_url=self._elevenlabs_base_url,
        )
        voice_card.start_cloning()
        voice_worker = VoiceCloneWorker(voice_config, self)
        voice_worker.finished_ok.connect(self._on_voice_done)
        voice_worker.failed.connect(self._on_voice_failed)
        self._voice_worker = voice_worker
        voice_worker.start()

    def _on_cancel_generation(self) -> None:
        if self._pixel_worker is not None:
            self._pixel_worker.request_cancel()
        if self._voice_worker is not None:
            self._voice_worker.request_cancel()

    def _on_pixel_done(self, sheet: Image.Image, meta: object, problems: list[str]) -> None:
        del meta
        self._pixel_result = (sheet, problems)
        self._maybe_finish()

    def _on_pixel_failed(self, message: str) -> None:
        self._pixel_worker = None
        if self._voice_worker is not None:
            self._voice_worker.request_cancel()
            self._voice_worker = None
        if self._progress_view is not None:
            self._progress_view.freeze()
        if self._voice_card is not None:
            self._voice_card.show_skipped("像素生成失败，已取消声音克隆")
        self._stack.setCurrentWidget(self._form_view)
        self._show_error(f"生成失败：{message}")

    def _on_pixel_cancelled(self) -> None:
        self._pixel_worker = None
        if self._voice_worker is not None:
            self._voice_worker.request_cancel()
            self._voice_worker = None
        if self._progress_view is not None:
            self._progress_view.freeze()
        if self._voice_card is not None:
            self._voice_card.show_skipped("已取消生成")
        self._stack.setCurrentWidget(self._form_view)
        self._show_error("已取消生成")

    def _on_voice_done(self, voice_id: str) -> None:
        self._voice_result = ("ok", voice_id)
        if self._voice_card is not None:
            self._voice_card.show_success()
        self._maybe_finish()

    def _on_voice_failed(self, message: str) -> None:
        self._voice_result = ("failed", message)
        if self._voice_card is not None:
            self._voice_card.show_failure(message)
        self._maybe_finish()

    def _maybe_finish(self) -> None:
        if self._pixel_result is None:
            return
        if self._voice_result is None and not self._voice_skipped:
            return

        sheet, problems = self._pixel_result
        output_dir = self._output_dir
        assert output_dir is not None

        voice_error: str | None = None
        if self._voice_result is not None:
            status, payload = self._voice_result
            if status == "ok":
                save_pet_voice_config(
                    output_dir,
                    PetVoiceConfig(provider=TTSProviderName.ELEVENLABS, voice=payload),
                )
            else:
                voice_error = payload

        self._pixel_worker = None
        self._voice_worker = None
        if self._progress_view is not None:
            self._progress_view.finish_success()
            if problems:
                self._progress_view.show_qa_warnings(problems)
        QTimer.singleShot(_GLOW_DURATION_MS, lambda: self._show_completion_page(sheet, voice_error))

    # ------------------------------------------------------------------
    # 第 5 页：完成
    # ------------------------------------------------------------------

    def _build_completion_page(self, sheet: Image.Image, voice_error: str | None) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)

        thumbnail_label = QLabel(page)
        thumbnail_label.setPixmap(_pil_to_pixmap(sheet, _COMPLETION_THUMBNAIL_SIZE))
        layout.addWidget(thumbnail_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        title_label = StrongBodyLabel("角色已创建！", page)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        notes: list[str] = []
        if self._voice_skipped:
            notes.append("未绑定专属声音，可以之后在画廊里为它单独更换声音。")
        elif voice_error is not None:
            notes.append(f"声音克隆失败（{voice_error}），可以之后在画廊里为它单独更换声音。")
        if self._settings is not None and not self._settings.tts.enabled:
            notes.append("如需听到声音，请在设置里打开语音合成总开关。")

        note_label = CaptionLabel("\n".join(notes), page)
        note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note_label.setWordWrap(True)
        note_label.setVisible(bool(notes))
        layout.addWidget(note_label)

        layout.addStretch(1)
        close_button = PrimaryPushButton("完成", page)
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        return page

    def _show_completion_page(self, sheet: Image.Image, voice_error: str | None) -> None:
        page = self._build_completion_page(sheet, voice_error)
        self._stack.addWidget(page)
        self._stack.setCurrentWidget(page)
        output_dir = self._output_dir
        if output_dir is not None:
            self.character_created.emit(output_dir)
