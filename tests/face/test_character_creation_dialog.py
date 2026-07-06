"""``CharacterCreationDialog`` 的表单校验、凭证预填与生成进度视图回归测试。

生成进度相关的槽函数（``_on_generation_finished``/``_on_generation_failed``/
``_on_generation_cancelled``）直接灌入合成参数调用，不真的启动
``CharacterGenerationWorker`` 线程，保证测试确定性。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QFileDialog
from qfluentwidgets import PushButton

from miku_on_desk.character_generation import STATE_SPECS, GenerationConfig, GenerationProgress
from miku_on_desk.config.settings import AppSettings, ImageGenerationConfig
from miku_on_desk.face.character_generation_worker import CharacterGenerationWorker
from miku_on_desk.face.ui.character_creation_dialog import (
    CharacterCreationDialog,
    _GenerationProgressView,
)


def _fill_valid_form(dialog: CharacterCreationDialog) -> None:
    dialog._name_edit.setText("new_pet")
    dialog._description_edit.setPlainText("a cool original cat character")
    dialog._api_key_edit.setText("sk-test")


def test_validate_form_rejects_invalid_name(qapp: QApplication, tmp_path: Path) -> None:
    dialog = CharacterCreationDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    dialog._name_edit.setText("bad name!")
    dialog._description_edit.setPlainText("desc")
    dialog._api_key_edit.setText("sk-test")

    config = dialog._validate_form()

    assert config is None
    assert not dialog._error_label.isHidden()


def test_validate_form_rejects_existing_output_dir(qapp: QApplication, tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    (assets_dir / "existing_pet").mkdir()
    dialog = CharacterCreationDialog(assets_dir, tmp_path / "settings.json")
    dialog._name_edit.setText("existing_pet")
    dialog._description_edit.setPlainText("desc")
    dialog._api_key_edit.setText("sk-test")

    config = dialog._validate_form()

    assert config is None
    assert not dialog._error_label.isHidden()


def test_validate_form_rejects_empty_description(qapp: QApplication, tmp_path: Path) -> None:
    dialog = CharacterCreationDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    dialog._name_edit.setText("new_pet")
    dialog._api_key_edit.setText("sk-test")

    config = dialog._validate_form()

    assert config is None
    assert not dialog._error_label.isHidden()


def test_validate_form_rejects_empty_api_key(qapp: QApplication, tmp_path: Path) -> None:
    dialog = CharacterCreationDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    dialog._name_edit.setText("new_pet")
    dialog._description_edit.setPlainText("desc")

    config = dialog._validate_form()

    assert config is None
    assert not dialog._error_label.isHidden()


def test_validate_form_returns_config_for_valid_input(qapp: QApplication, tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets_pets"
    dialog = CharacterCreationDialog(assets_dir, tmp_path / "settings.json")
    _fill_valid_form(dialog)
    dialog._base_url_edit.setText("https://example.com/v1")
    dialog._model_combo.setCurrentText("gpt-image-2")

    config = dialog._validate_form()

    assert config is not None
    assert config.pet_name == "new_pet"
    assert config.description == "a cool original cat character"
    assert config.output_dir == assets_dir / "new_pet"
    assert config.api_key == "sk-test"
    assert config.base_url == "https://example.com/v1"
    assert config.model == "gpt-image-2"
    assert dialog._error_label.isHidden()


def test_prefill_from_settings_populates_credentials(qapp: QApplication, tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.image_generation = ImageGenerationConfig(
        api_key="sk-existing", base_url="https://example.com/v1", model="gpt-image-2"
    )
    settings.save(settings_path)

    dialog = CharacterCreationDialog(tmp_path / "assets_pets", settings_path)

    assert dialog._base_url_edit.text() == "https://example.com/v1"
    assert dialog._api_key_edit.text() == "sk-existing"
    assert dialog._model_combo.currentText() == "gpt-image-2"


def test_on_browse_reference_image_sets_path_and_label(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = CharacterCreationDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    chosen = tmp_path / "ref.png"
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_a, **_k: (str(chosen), ""))

    dialog._on_browse_reference_image()

    assert dialog._reference_image_path == chosen
    assert dialog._reference_label.text() == "ref.png"


def test_on_browse_reference_image_ignores_cancelled_dialog(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = CharacterCreationDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_a, **_k: ("", ""))

    dialog._on_browse_reference_image()

    assert dialog._reference_image_path is None


def test_close_button_is_left_of_start_button_and_closes_dialog(
    qapp: QApplication, tmp_path: Path
) -> None:
    dialog = CharacterCreationDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    dialog.show()
    buttons = dialog._form_view.findChildren(PushButton)
    close_buttons = [button for button in buttons if button.text() == "关闭"]
    assert len(close_buttons) == 1
    close_button = close_buttons[0]

    close_button.click()

    assert dialog.isHidden()


def test_generation_progress_view_on_progress_reference_stage_sets_status(
    qapp: QApplication,
) -> None:
    view = _GenerationProgressView(4, 4)
    progress = GenerationProgress(
        stage="reference", detail="", completed_states=0, total_states=len(STATE_SPECS)
    )

    view.on_progress(progress)

    assert view._status_label.text() == "生成基准参考图…"


def test_generation_progress_view_on_progress_strip_stage_updates_progress_and_tiles(
    qapp: QApplication,
) -> None:
    view = _GenerationProgressView(4, 4)
    strip_image = Image.new("RGBA", (8, 8), (10, 20, 30, 255))
    first_state = STATE_SPECS[0].state
    progress = GenerationProgress(
        stage="strip",
        detail=first_state.value,
        completed_states=1,
        total_states=len(STATE_SPECS),
        strip_image=strip_image,
    )

    view.on_progress(progress)

    assert view._progress_bar.value() == 1
    assert view._reference_done is True
    assert view._reference_tile._image_label.styleSheet() == view._reference_tile._done_style
    assert view._state_tiles[first_state]._image_label.styleSheet() == (
        view._state_tiles[first_state]._done_style
    )


def test_generation_progress_view_on_progress_assemble_stage_sets_status(
    qapp: QApplication,
) -> None:
    view = _GenerationProgressView(4, 4)
    progress = GenerationProgress(
        stage="assemble",
        detail="",
        completed_states=len(STATE_SPECS),
        total_states=len(STATE_SPECS),
    )

    view.on_progress(progress)

    assert view._status_label.text() == "拼装 spritesheet…"


def test_generation_progress_view_on_progress_qa_stage_sets_status(qapp: QApplication) -> None:
    view = _GenerationProgressView(4, 4)
    progress = GenerationProgress(
        stage="qa", detail="", completed_states=len(STATE_SPECS), total_states=len(STATE_SPECS)
    )

    view.on_progress(progress)

    assert view._status_label.text() == "运行 QA 检查…"


def test_generation_progress_view_finish_success_freezes_and_completes_bar(
    qapp: QApplication,
) -> None:
    view = _GenerationProgressView(4, 4)

    view.finish_success()

    assert view._status_label.text() == "生成完成！"
    assert view._progress_bar.value() == view._progress_bar.maximum()
    assert view._cancel_button.isEnabled() is False


def test_on_generation_finished_emits_character_created_and_freezes_progress_view(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: fn())
    assets_dir = tmp_path / "assets_pets"
    dialog = CharacterCreationDialog(assets_dir, tmp_path / "settings.json")
    output_dir = assets_dir / "new_pet"
    dialog._output_dir = output_dir
    progress_view = _GenerationProgressView(4, 4)
    dialog._progress_view = progress_view
    config = GenerationConfig(
        pet_name="new_pet", description="d", output_dir=output_dir, api_key="sk-test"
    )
    dialog._worker = CharacterGenerationWorker(config, dialog)
    created: list[Path] = []
    dialog.character_created.connect(created.append)

    dialog._on_generation_finished(Image.new("RGBA", (4, 4)), object(), [])

    assert dialog._worker is None
    assert created == [output_dir]
    assert progress_view._status_label.text() == "生成完成！"


def test_on_generation_failed_shows_error_and_returns_to_form_view(
    qapp: QApplication, tmp_path: Path
) -> None:
    dialog = CharacterCreationDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    progress_view = _GenerationProgressView(4, 4)
    dialog._progress_view = progress_view
    config = GenerationConfig(
        pet_name="new_pet", description="d", output_dir=tmp_path / "new_pet", api_key="sk-test"
    )
    dialog._worker = CharacterGenerationWorker(config, dialog)

    dialog._on_generation_failed("API 超时")

    assert dialog._worker is None
    assert dialog._stack.currentWidget() is dialog._form_view
    assert dialog._error_label.text() == "生成失败：API 超时"
    assert not dialog._error_label.isHidden()
    assert progress_view._cancel_button.isEnabled() is False


def test_on_generation_cancelled_shows_error_and_returns_to_form_view(
    qapp: QApplication, tmp_path: Path
) -> None:
    dialog = CharacterCreationDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    progress_view = _GenerationProgressView(4, 4)
    dialog._progress_view = progress_view
    config = GenerationConfig(
        pet_name="new_pet", description="d", output_dir=tmp_path / "new_pet", api_key="sk-test"
    )
    dialog._worker = CharacterGenerationWorker(config, dialog)

    dialog._on_generation_cancelled()

    assert dialog._worker is None
    assert dialog._stack.currentWidget() is dialog._form_view
    assert dialog._error_label.text() == "已取消生成"
    assert progress_view._cancel_button.isEnabled() is False
