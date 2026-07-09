"""创建新角色弹窗：上传参考图（可选）+ 文字描述 + 图像生成 API 凭证，启动后台生成，
并用一排小格展示每个阶段的呼吸等待动效，逐格替换成生成好的真实缩略图。

用 ``QWidget`` 而非 ``QDialog``、``setWindowFlags(Tool | FramelessWindowHint)``：
与 ``chat_popup.py`` 相同的 QTBUG-83490 规避——``Popup`` 类型窗口在 macOS/Windows 上
成不了正常的"key window"，会破坏中文输入法候选词上屏，描述输入框必须避开这个坑。
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
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
    ListWidget,
    PlainTextEdit,
    PrimaryPushButton,
    ProgressBar,
    PushButton,
    StrongBodyLabel,
)

from miku_on_desk.character_generation import (
    STATE_SPECS,
    GenerationConfig,
    GenerationProgress,
)
from miku_on_desk.config.settings import (
    AppSettings,
    load_settings_with_vault,
    save_settings_with_vault,
)
from miku_on_desk.face.character_generation_worker import CharacterGenerationWorker
from miku_on_desk.face.pet_state import PetState
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

if TYPE_CHECKING:
    from miku_on_desk.brain.secrets.vault import SecretVault

_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
_MODEL_CHOICES = ("gpt-image-1", "gpt-image-2")
_TILE_SIZE = 64
_THUMBNAIL_SIZE = 48
_BREATH_DURATION_MS = 1200
_RESULT_FADE_MS = 200
_GLOW_DURATION_MS = 900
_GLOW_FADE_MS = 300


def _pil_to_pixmap(image: Image.Image, size: int) -> QPixmap:
    buffer = io.BytesIO()
    image.convert("RGBA").save(buffer, format="PNG")
    pixmap = QPixmap()
    # 不传 format：显式传 b"PNG" 在这个 PySide6 版本上会在运行时抛
    # "wrong argument values"（尽管类型签名匹配），省略参数让 Qt 自动嗅探反而能跑通。
    pixmap.loadFromData(buffer.getvalue())
    return pixmap.scaled(
        size,
        size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )


class _GenerationTile(QWidget):
    """单个生成阶段（参考图或某状态）的小格：灰色剪影呼吸循环 → 完成后淡入真实缩略图。"""

    def __init__(self, label_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_XXS, SPACING_XXS, SPACING_XXS, SPACING_XXS)

        self._image_label = StrongBodyLabel("", self)
        self._image_label.setFixedSize(_TILE_SIZE, _TILE_SIZE)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._idle_style = f"background-color: {PLACEHOLDER_BG}; border-radius: {RADIUS_MD}px;"
        self._done_style = (
            f"background-color: {PLACEHOLDER_BG}; {border_qss(TEAL_DARK, radius=RADIUS_MD)}"
        )
        self._image_label.setStyleSheet(self._idle_style)
        layout.addWidget(self._image_label)

        caption = CaptionLabel(label_text, self)
        caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(caption)

        self._opacity_effect = QGraphicsOpacityEffect(self._image_label)
        self._opacity_effect.setOpacity(1.0)
        self._image_label.setGraphicsEffect(self._opacity_effect)

        self._breath_anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._breath_anim.setDuration(_BREATH_DURATION_MS)
        self._breath_anim.setKeyValueAt(0.0, 0.35)
        self._breath_anim.setKeyValueAt(0.5, 1.0)
        self._breath_anim.setKeyValueAt(1.0, 0.35)
        self._breath_anim.setLoopCount(-1)

    def start_breathing(self) -> None:
        self._breath_anim.start()

    def stop_breathing(self) -> None:
        self._breath_anim.stop()
        self._opacity_effect.setOpacity(1.0)

    def mark_done_placeholder(self) -> None:
        """没有真实缩略图可展示时（目前参考图阶段就是这种情况），只做完成态高亮。"""
        self.stop_breathing()
        self._image_label.setStyleSheet(self._done_style)

    def show_result(self, image: Image.Image) -> None:
        self.stop_breathing()
        self._image_label.setStyleSheet(self._done_style)
        self._image_label.setPixmap(_pil_to_pixmap(image, _TILE_SIZE))

        fade = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        fade.setDuration(_RESULT_FADE_MS)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        fade.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        self._fade_anim = fade


class _GenerationProgressView(QWidget):
    """生成中视图：顶部进度条 + 状态文字，中间一排呼吸/完成动效小格，底部取消按钮。"""

    cancel_requested = Signal()

    def __init__(self, frame_width: int, frame_height: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._reference_done = False

        layout = QVBoxLayout(self)
        self._status_label = StrongBodyLabel("准备生成…", self)
        layout.addWidget(self._status_label)

        self._progress_bar = ProgressBar(self)
        self._progress_bar.setRange(0, len(STATE_SPECS))
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        tiles_row = QHBoxLayout()
        self._reference_tile = _GenerationTile("参考图", self)
        tiles_row.addWidget(self._reference_tile)
        self._state_tiles: dict[PetState, _GenerationTile] = {}
        for spec in STATE_SPECS:
            tile = _GenerationTile(spec.state.value, self)
            tiles_row.addWidget(tile)
            self._state_tiles[spec.state] = tile
        layout.addLayout(tiles_row)

        self._qa_list = ListWidget(self)
        self._qa_list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._qa_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._qa_list.hide()
        layout.addWidget(self._qa_list)

        self._cancel_button = PushButton("取消生成", self)
        self._cancel_button.clicked.connect(self.cancel_requested)
        layout.addWidget(self._cancel_button)

        self._reference_tile.start_breathing()

    def on_progress(self, progress: GenerationProgress) -> None:
        if progress.stage == "reference":
            if progress.reference_image is not None:
                if not self._reference_done:
                    self._reference_tile.show_result(progress.reference_image)
                    self._on_reference_done()
                return
            self._status_label.setText("生成基准参考图…")
            return
        if progress.stage == "strip":
            self._status_label.setText(
                f"生成状态动画 {progress.completed_states}/{progress.total_states}："
                f"{progress.detail}"
            )
            self._progress_bar.setValue(progress.completed_states)
            if not self._reference_done:
                self._reference_tile.mark_done_placeholder()
                self._on_reference_done()
            # 各状态并发生成，完成顺序和 STATE_SPECS 顺序无关，只能靠 detail（该状态自己的
            # 名字）认领对应的 tile，不能像以前那样假设 completed_states 就是下标。
            completed_state = PetState(progress.detail)
            if progress.strip_image is not None:
                frame = progress.strip_image.crop((0, 0, self._frame_width, self._frame_height))
                self._state_tiles[completed_state].show_result(frame)
            return
        if progress.stage == "assemble":
            self._status_label.setText("拼装 spritesheet…")
            return
        self._status_label.setText("运行 QA 检查…")

    def _on_reference_done(self) -> None:
        """参考图就位后所有状态动作条几乎同时开始并发请求，对应 tile 一起进入呼吸态。"""
        self._reference_done = True
        for tile in self._state_tiles.values():
            tile.start_breathing()

    def freeze(self) -> None:
        self._reference_tile.stop_breathing()
        for tile in self._state_tiles.values():
            tile.stop_breathing()
        self._cancel_button.setEnabled(False)

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def show_qa_warnings(self, problems: list[str]) -> None:
        if not problems:
            return
        self._status_label.setText("生成完成，但 QA 有提示：")
        self._qa_list.clear()
        self._qa_list.addItems(problems)
        self._qa_list.setFixedHeight(min(120, 24 * len(problems) + 8))
        self._qa_list.show()

    def finish_success(self) -> None:
        self.freeze()
        self._status_label.setText("生成完成！")
        self._progress_bar.setValue(self._progress_bar.maximum())
        self.setStyleSheet(f"_GenerationProgressView {{ {border_qss(TEAL_MAIN)} }}")
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
            f"_GenerationProgressView {{ border: 2px solid "
            f"rgba({color.red()}, {color.green()}, {color.blue()}, {alpha}); "
            f"border-radius: {RADIUS_LG}px; }}"
        )


class CharacterCreationDialog(QWidget):
    """两段式内容：表单视图 → 提交后切到生成中视图，全程同一个窗口，不额外弹第二个窗口。"""

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
        self.setWindowTitle("创建新角色")
        self.resize(560, 420)
        self._assets_pets_dir = assets_pets_dir
        self._settings_path = settings_path
        self._vault = vault
        self._reference_image_path: Path | None = None
        self._worker: CharacterGenerationWorker | None = None
        self._progress_view: _GenerationProgressView | None = None
        self._output_dir: Path | None = None

        self._stack = QStackedLayout(self)
        self._form_view = self._build_form_view()
        self._stack.addWidget(self._form_view)
        self._prefill_from_settings()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self._worker is None:
            self.close()
            return
        super().keyPressEvent(event)

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

        reference_row = QHBoxLayout()
        self._reference_label = CaptionLabel("未选择（可选）", container)
        reference_row.addWidget(self._reference_label)
        reference_button = PushButton("选择参考图…", container)
        reference_button.clicked.connect(self._on_browse_reference_image)
        reference_row.addWidget(reference_button)
        self._reference_thumbnail = QLabel(container)
        self._reference_thumbnail.setFixedSize(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE)
        self._reference_thumbnail.hide()
        reference_row.addWidget(self._reference_thumbnail)
        form.addRow("参考图", reference_row)

        self._description_edit = PlainTextEdit(container)
        self._description_edit.setPlaceholderText("用文字描述这个角色的外观、服装、配色…")
        form.addRow("角色描述", self._description_edit)

        self._base_url_edit = LineEdit(container)
        self._base_url_edit.setPlaceholderText("默认官方地址")
        form.addRow("API Base URL", self._base_url_edit)

        self._api_key_edit = LineEdit(container)
        self._api_key_edit.setEchoMode(LineEdit.EchoMode.Password)
        form.addRow("API Key", self._api_key_edit)

        self._model_combo = ComboBox(container)
        self._model_combo.addItems(list(_MODEL_CHOICES))
        form.addRow("模型", self._model_combo)

        layout.addLayout(form)
        layout.addStretch(1)

        button_row = QHBoxLayout()
        close_button = PushButton("关闭", container)
        close_button.clicked.connect(self.close)
        button_row.addWidget(close_button)
        start_button = PrimaryPushButton("开始生成", container)
        start_button.clicked.connect(self._on_start_generation)
        button_row.addWidget(start_button)
        layout.addLayout(button_row)

        return container

    def _prefill_from_settings(self) -> None:
        if self._vault is not None:
            settings = load_settings_with_vault(self._settings_path, self._vault)
        else:
            settings = AppSettings.load(self._settings_path)
        image_generation = settings.image_generation
        self._base_url_edit.setText(image_generation.base_url or "")
        self._api_key_edit.setText(image_generation.api_key or "")
        if image_generation.model in _MODEL_CHOICES:
            self._model_combo.setCurrentText(image_generation.model)

    def _on_browse_reference_image(self) -> None:
        path_str, _filter = QFileDialog.getOpenFileName(
            self, "选择参考图", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not path_str:
            return
        self._reference_image_path = Path(path_str)
        self._reference_label.setText(self._reference_image_path.name)
        pixmap = QPixmap(str(self._reference_image_path)).scaled(
            _THUMBNAIL_SIZE,
            _THUMBNAIL_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if not pixmap.isNull():
            self._reference_thumbnail.setPixmap(pixmap)
            self._reference_thumbnail.show()

    def _show_error(self, message: str) -> None:
        self._error_label.setText(message)
        self._error_label.show()

    def _validate_form(self) -> GenerationConfig | None:
        name = self._name_edit.text().strip()
        if not _NAME_PATTERN.match(name):
            self._show_error("角色名称只能包含字母、数字、下划线、短横线，且不能为空")
            return None
        output_dir = self._assets_pets_dir / name
        if output_dir.exists():
            self._show_error(f"角色名称 “{name}” 已存在，请换一个")
            return None

        description = self._description_edit.toPlainText().strip()
        if not description:
            self._show_error("请填写角色描述")
            return None

        api_key = self._api_key_edit.text().strip()
        if not api_key:
            self._show_error("请填写 API Key")
            return None

        self._error_label.hide()
        return GenerationConfig(
            pet_name=name,
            description=description,
            output_dir=output_dir,
            model=self._model_combo.currentText(),
            api_key=api_key,
            base_url=self._base_url_edit.text().strip() or None,
            reference_image_path=self._reference_image_path,
        )

    def _on_start_generation(self) -> None:
        config = self._validate_form()
        if config is None:
            return
        self._output_dir = config.output_dir

        settings = AppSettings.load(self._settings_path)
        settings.image_generation.api_key = config.api_key
        settings.image_generation.base_url = config.base_url
        settings.image_generation.model = config.model
        if self._vault is not None:
            save_settings_with_vault(settings, self._settings_path, self._vault)
        else:
            settings.save(self._settings_path)

        progress_view = _GenerationProgressView(config.frame_width, config.frame_height, self)
        progress_view.cancel_requested.connect(self._on_cancel_generation)
        self._progress_view = progress_view
        self._stack.addWidget(progress_view)
        self._stack.setCurrentWidget(progress_view)

        worker = CharacterGenerationWorker(config, self)
        worker.progress.connect(progress_view.on_progress)
        worker.finished_ok.connect(self._on_generation_finished)
        worker.failed.connect(self._on_generation_failed)
        worker.cancelled.connect(self._on_generation_cancelled)
        self._worker = worker
        worker.start()

    def _on_cancel_generation(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()

    def _on_generation_finished(
        self, sheet: Image.Image, meta: object, problems: list[str]
    ) -> None:
        del sheet, meta
        self._worker = None
        output_dir = self._output_dir
        if self._progress_view is not None:
            self._progress_view.finish_success()
            if problems:
                self._progress_view.show_qa_warnings(problems)
        if output_dir is not None:
            QTimer.singleShot(_GLOW_DURATION_MS, lambda: self.character_created.emit(output_dir))

    def _on_generation_failed(self, message: str) -> None:
        self._worker = None
        if self._progress_view is not None:
            self._progress_view.freeze()
        self._stack.setCurrentWidget(self._form_view)
        self._show_error(f"生成失败：{message}")

    def _on_generation_cancelled(self) -> None:
        self._worker = None
        if self._progress_view is not None:
            self._progress_view.freeze()
        self._stack.setCurrentWidget(self._form_view)
        self._show_error("已取消生成")
