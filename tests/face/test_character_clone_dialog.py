"""``CharacterCloneDialog`` 的表单校验、分页流转与并行 join 逻辑回归测试。

跟 ``test_character_creation_dialog.py`` 同款手法：真正会碰线程/网络/硬件的私有方法
（``_show_generation_page``/``_show_reading_recording_page``）在对应测试里被整个替换掉，
真正需要验证 join 逻辑的地方则构造一个真实但从未 ``.start()`` 的 worker，直接调私有
完成回调方法模拟其终态。摄像头相关测试统一 monkeypatch ``QMediaDevices.videoInputs``/
``audioInputs`` 为空，避免在真的装了摄像头的机器上触发系统权限弹窗。
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from qfluentwidgets import CaptionLabel

from miku_on_desk.brain.tts.voice_clone import VoiceCloneConfig
from miku_on_desk.character_generation import GenerationConfig
from miku_on_desk.config.settings import (
    AppSettings,
    ImageGenerationConfig,
    TTSProviderName,
    VoiceCloningConfig,
)
from miku_on_desk.face.character_generation_worker import CharacterGenerationWorker
from miku_on_desk.face.character_voice import load_pet_voice_config
from miku_on_desk.face.ui import capture_widgets
from miku_on_desk.face.ui.character_clone_dialog import (
    _DEFAULT_DESCRIPTION,
    CharacterCloneDialog,
    _VoiceCloneStatusCard,
)
from miku_on_desk.face.ui.character_creation_dialog import _GenerationProgressView
from miku_on_desk.face.voice_clone_worker import VoiceCloneWorker


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (10, 10), (255, 0, 0, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _force_no_camera_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capture_widgets.QMediaDevices, "videoInputs", staticmethod(lambda: []))
    monkeypatch.setattr(capture_widgets.QMediaDevices, "audioInputs", staticmethod(lambda: []))


def _fill_valid_form(dialog: CharacterCloneDialog) -> None:
    dialog._name_edit.setText("new_pet")
    dialog._api_key_edit.setText("sk-test")


def test_validate_form_rejects_invalid_name(qapp: QApplication, tmp_path: Path) -> None:
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    dialog._name_edit.setText("../evil")
    dialog._api_key_edit.setText("sk-test")

    result = dialog._validate_form()

    assert result is None
    assert not dialog._error_label.isHidden()


def test_validate_form_rejects_existing_output_dir(qapp: QApplication, tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets_pets"
    (assets_dir / "new_pet").mkdir(parents=True)
    dialog = CharacterCloneDialog(assets_dir, tmp_path / "settings.json")
    _fill_valid_form(dialog)

    result = dialog._validate_form()

    assert result is None
    assert not dialog._error_label.isHidden()


def test_validate_form_rejects_empty_api_key(qapp: QApplication, tmp_path: Path) -> None:
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    dialog._name_edit.setText("new_pet")

    result = dialog._validate_form()

    assert result is None
    assert not dialog._error_label.isHidden()


def test_validate_form_uses_default_description_when_blank(
    qapp: QApplication, tmp_path: Path
) -> None:
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    _fill_valid_form(dialog)

    config = dialog._validate_form()

    assert config is not None
    assert config.description == _DEFAULT_DESCRIPTION


def test_validate_form_returns_config_for_valid_input(qapp: QApplication, tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets_pets"
    dialog = CharacterCloneDialog(assets_dir, tmp_path / "settings.json")
    _fill_valid_form(dialog)
    dialog._description_edit.setPlainText("一个爱笑的猫娘")

    config = dialog._validate_form()

    assert config is not None
    assert config.pet_name == "new_pet"
    assert config.description == "一个爱笑的猫娘"
    assert config.output_dir == assets_dir / "new_pet"
    assert config.api_key == "sk-test"


def test_prefill_from_settings_loads_image_and_voice_cloning_credentials(
    qapp: QApplication, tmp_path: Path
) -> None:
    settings_path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.image_generation = ImageGenerationConfig(
        api_key="sk-image", base_url="https://img.example.com", model="gpt-image-1"
    )
    settings.voice_cloning = VoiceCloningConfig(
        elevenlabs_api_key="el-key", elevenlabs_base_url="https://el.example.com"
    )
    settings.save(settings_path)

    dialog = CharacterCloneDialog(tmp_path / "assets_pets", settings_path)

    assert dialog._api_key_edit.text() == "sk-image"
    assert dialog._base_url_edit.text() == "https://img.example.com"
    assert dialog._model_combo.currentText() == "gpt-image-1"
    assert dialog._settings is not None
    assert dialog._settings.voice_cloning.elevenlabs_api_key == "el-key"
    assert dialog._settings.voice_cloning.elevenlabs_base_url == "https://el.example.com"


def test_on_form_next_clicked_persists_image_settings_and_advances_to_photo_page(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_no_camera_devices(monkeypatch)
    settings_path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.voice_cloning = VoiceCloningConfig(elevenlabs_api_key="el-key")
    settings.save(settings_path)
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", settings_path)
    _fill_valid_form(dialog)

    dialog._on_form_next_clicked()

    assert dialog._stack.currentWidget() is not dialog._form_view
    saved = AppSettings.load(settings_path)
    assert saved.image_generation.api_key == "sk-test"
    assert saved.voice_cloning.elevenlabs_api_key == "el-key"


def test_show_photo_page_reports_unavailable_when_no_camera_device(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_no_camera_devices(monkeypatch)
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    _fill_valid_form(dialog)

    dialog._on_form_next_clicked()

    assert not dialog._photo_error_label.isHidden()
    assert not dialog._capture_button.isEnabled()
    assert dialog._photo_next_button.text() == "跳过拍照，仅用文字描述生成"
    assert dialog._photo_next_button.isEnabled()


def test_on_photo_captured_stores_bytes_and_enables_preview(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_no_camera_devices(monkeypatch)
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    _fill_valid_form(dialog)
    dialog._on_form_next_clicked()
    png_bytes = _png_bytes()

    dialog._on_photo_captured(png_bytes)

    assert dialog._captured_photo_bytes == png_bytes
    assert dialog._photo_next_button.isEnabled()
    assert not dialog._photo_preview_label.isHidden()


def test_on_photo_next_clicked_stops_camera_and_advances(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_no_camera_devices(monkeypatch)
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    _fill_valid_form(dialog)
    dialog._on_form_next_clicked()
    advanced: list[bool] = []
    monkeypatch.setattr(dialog, "_show_reading_recording_page", lambda: advanced.append(True))

    dialog._on_photo_next_clicked()

    assert dialog._camera_widget is None
    assert advanced == [True]


def test_on_recording_recorded_stores_bytes_and_advances(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    advanced: list[bool] = []
    monkeypatch.setattr(dialog, "_show_generation_page", lambda: advanced.append(True))

    dialog._on_recording_recorded(b"wav-bytes")

    assert dialog._recorded_wav_bytes == b"wav-bytes"
    assert dialog._reading_recording_widget is None
    assert advanced == [True]


def test_on_recording_skipped_clears_bytes_and_advances(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    dialog._recorded_wav_bytes = b"stale-bytes"
    advanced: list[bool] = []
    monkeypatch.setattr(dialog, "_show_generation_page", lambda: advanced.append(True))

    dialog._on_recording_skipped()

    assert dialog._recorded_wav_bytes is None
    assert advanced == [True]


def test_show_generation_page_reads_elevenlabs_credentials_from_settings(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(CharacterGenerationWorker, "start", lambda self: None)
    monkeypatch.setattr(VoiceCloneWorker, "start", lambda self: None)

    settings_path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.voice_cloning = VoiceCloningConfig(
        elevenlabs_api_key="el-key", elevenlabs_base_url="https://el.example.com"
    )
    settings.save(settings_path)

    assets_dir = tmp_path / "assets_pets"
    dialog = CharacterCloneDialog(assets_dir, settings_path)
    dialog._generation_config = GenerationConfig(
        pet_name="new_pet", description="d", output_dir=assets_dir / "new_pet", api_key="sk-test"
    )
    dialog._recorded_wav_bytes = b"wav-bytes"

    dialog._show_generation_page()

    assert dialog._voice_skipped is False
    assert dialog._voice_worker is not None
    assert dialog._voice_worker._config.api_key == "el-key"
    assert dialog._voice_worker._config.base_url == "https://el.example.com"


def test_show_generation_page_skips_voice_when_settings_has_no_elevenlabs_key(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(CharacterGenerationWorker, "start", lambda self: None)

    assets_dir = tmp_path / "assets_pets"
    dialog = CharacterCloneDialog(assets_dir, tmp_path / "settings.json")
    dialog._generation_config = GenerationConfig(
        pet_name="new_pet", description="d", output_dir=assets_dir / "new_pet", api_key="sk-test"
    )
    dialog._recorded_wav_bytes = b"wav-bytes"

    dialog._show_generation_page()

    assert dialog._voice_skipped is True
    assert dialog._voice_worker is None


def test_config_with_reference_photo_writes_temp_file_and_sets_selfie_kind(
    qapp: QApplication, tmp_path: Path
) -> None:
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    config = GenerationConfig(
        pet_name="new_pet", description="d", output_dir=tmp_path / "new_pet", api_key="sk-test"
    )
    png_bytes = _png_bytes()

    updated = dialog._config_with_reference_photo(config, png_bytes)

    assert updated.reference_image_kind == "selfie"
    assert updated.reference_image_path is not None
    assert updated.reference_image_path.read_bytes() == png_bytes


def test_on_cancel_generation_requests_cancel_on_both_workers(
    qapp: QApplication, tmp_path: Path
) -> None:
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    config = GenerationConfig(
        pet_name="new_pet", description="d", output_dir=tmp_path / "new_pet", api_key="sk-test"
    )
    pixel_worker = CharacterGenerationWorker(config, dialog)
    dialog._pixel_worker = pixel_worker
    voice_config = VoiceCloneConfig(name="new_pet", audio_bytes=b"wav", api_key="sk-el")
    voice_worker = VoiceCloneWorker(voice_config, dialog)
    dialog._voice_worker = voice_worker

    dialog._on_cancel_generation()

    assert pixel_worker._cancel_requested.is_set()
    assert voice_worker._cancel_requested.is_set()


def test_maybe_finish_waits_for_voice_result_when_not_skipped(
    qapp: QApplication, tmp_path: Path
) -> None:
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    dialog._voice_skipped = False
    created: list[Path] = []
    dialog.character_created.connect(created.append)

    dialog._on_pixel_done(Image.new("RGBA", (4, 4)), object(), [])

    assert created == []
    assert dialog._pixel_result is not None


def test_maybe_finish_completes_and_saves_voice_config_when_both_succeed(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: fn())
    assets_dir = tmp_path / "assets_pets"
    dialog = CharacterCloneDialog(assets_dir, tmp_path / "settings.json")
    output_dir = assets_dir / "new_pet"
    output_dir.mkdir(parents=True)
    dialog._output_dir = output_dir
    dialog._voice_skipped = False
    progress_view = _GenerationProgressView(4, 4)
    dialog._progress_view = progress_view
    voice_card = _VoiceCloneStatusCard()
    dialog._voice_card = voice_card
    config = GenerationConfig(
        pet_name="new_pet", description="d", output_dir=output_dir, api_key="sk-test"
    )
    dialog._pixel_worker = CharacterGenerationWorker(config, dialog)
    voice_config = VoiceCloneConfig(name="new_pet", audio_bytes=b"wav", api_key="sk-el")
    dialog._voice_worker = VoiceCloneWorker(voice_config, dialog)
    created: list[Path] = []
    dialog.character_created.connect(created.append)

    dialog._on_pixel_done(Image.new("RGBA", (4, 4)), object(), [])
    dialog._on_voice_done("voice-123")

    assert created == [output_dir]
    assert dialog._pixel_worker is None
    assert dialog._voice_worker is None
    saved = load_pet_voice_config(output_dir)
    assert saved is not None
    assert saved.provider == TTSProviderName.ELEVENLABS
    assert saved.voice == "voice-123"
    assert progress_view._status_label.text() == "生成完成！"


def test_maybe_finish_completes_without_saving_voice_config_when_voice_fails(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: fn())
    assets_dir = tmp_path / "assets_pets"
    dialog = CharacterCloneDialog(assets_dir, tmp_path / "settings.json")
    output_dir = assets_dir / "new_pet"
    output_dir.mkdir(parents=True)
    dialog._output_dir = output_dir
    dialog._voice_skipped = False
    dialog._progress_view = _GenerationProgressView(4, 4)
    dialog._voice_card = _VoiceCloneStatusCard()
    created: list[Path] = []
    dialog.character_created.connect(created.append)

    dialog._on_pixel_done(Image.new("RGBA", (4, 4)), object(), [])
    dialog._on_voice_failed("配额不足")

    assert created == [output_dir]
    assert load_pet_voice_config(output_dir) is None


def test_maybe_finish_completes_without_waiting_for_voice_when_skipped(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: fn())
    assets_dir = tmp_path / "assets_pets"
    dialog = CharacterCloneDialog(assets_dir, tmp_path / "settings.json")
    output_dir = assets_dir / "new_pet"
    output_dir.mkdir(parents=True)
    dialog._output_dir = output_dir
    dialog._voice_skipped = True
    created: list[Path] = []
    dialog.character_created.connect(created.append)

    dialog._on_pixel_done(Image.new("RGBA", (4, 4)), object(), [])

    assert created == [output_dir]
    assert load_pet_voice_config(output_dir) is None


def test_on_pixel_failed_cancels_voice_worker_and_returns_to_form_view(
    qapp: QApplication, tmp_path: Path
) -> None:
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    dialog._progress_view = _GenerationProgressView(4, 4)
    voice_card = _VoiceCloneStatusCard()
    dialog._voice_card = voice_card
    voice_config = VoiceCloneConfig(name="new_pet", audio_bytes=b"wav", api_key="sk-el")
    voice_worker = VoiceCloneWorker(voice_config, dialog)
    dialog._voice_worker = voice_worker

    dialog._on_pixel_failed("API 超时")

    assert voice_worker._cancel_requested.is_set()
    assert dialog._voice_worker is None
    assert dialog._pixel_worker is None
    assert dialog._stack.currentWidget() is dialog._form_view
    assert not dialog._error_label.isHidden()
    assert dialog._error_label.text() == "生成失败：API 超时"
    assert voice_card._status_label.text() == "像素生成失败，已取消声音克隆"


def test_on_pixel_cancelled_shows_error_and_returns_to_form_view(
    qapp: QApplication, tmp_path: Path
) -> None:
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    dialog._progress_view = _GenerationProgressView(4, 4)
    voice_card = _VoiceCloneStatusCard()
    dialog._voice_card = voice_card

    dialog._on_pixel_cancelled()

    assert dialog._pixel_worker is None
    assert dialog._stack.currentWidget() is dialog._form_view
    assert dialog._error_label.text() == "已取消生成"
    assert voice_card._status_label.text() == "已取消生成"


def test_build_completion_page_shows_skipped_note_when_voice_skipped(
    qapp: QApplication, tmp_path: Path
) -> None:
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    dialog._voice_skipped = True

    page = dialog._build_completion_page(Image.new("RGBA", (4, 4)), None)

    note_label = page.findChildren(CaptionLabel)[0]
    assert "未绑定专属声音" in note_label.text()
    assert not note_label.isHidden()


def test_build_completion_page_shows_voice_error_note_when_voice_failed(
    qapp: QApplication, tmp_path: Path
) -> None:
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    dialog._voice_skipped = False

    page = dialog._build_completion_page(Image.new("RGBA", (4, 4)), "配额不足")

    note_label = page.findChildren(CaptionLabel)[0]
    assert "声音克隆失败（配额不足）" in note_label.text()
    assert not note_label.isHidden()


def test_build_completion_page_hides_note_when_voice_ok_and_tts_enabled(
    qapp: QApplication, tmp_path: Path
) -> None:
    dialog = CharacterCloneDialog(tmp_path / "assets_pets", tmp_path / "settings.json")
    dialog._voice_skipped = False
    assert dialog._settings is not None
    dialog._settings.tts.enabled = True

    page = dialog._build_completion_page(Image.new("RGBA", (4, 4)), None)

    note_label = page.findChildren(CaptionLabel)[0]
    assert note_label.isHidden()
