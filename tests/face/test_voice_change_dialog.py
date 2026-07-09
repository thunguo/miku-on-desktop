"""``VoiceChangeDialog`` 的模式选择、录音克隆分支与手填分支回归测试。

跟 ``test_character_clone_dialog.py`` 同款手法：``_show_cloning_page`` 里真正会调
``worker.start()`` 起线程发网络请求的分支，测试里把 ``VoiceCloneWorker.start`` 换成
空实现，只验证信号路由与状态卡片文案，不真的起线程/连 ElevenLabs。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from miku_on_desk.config.settings import TTSProviderName, VoiceCloningConfig
from miku_on_desk.face.character_voice import (
    PetVoiceConfig,
    load_pet_voice_config,
    save_pet_voice_config,
)
from miku_on_desk.face.ui.voice_change_dialog import VoiceChangeDialog
from miku_on_desk.face.voice_clone_worker import VoiceCloneWorker


def _make_pet_dir(tmp_path: Path) -> Path:
    pet_dir = tmp_path / "pet_a"
    pet_dir.mkdir()
    return pet_dir


def test_on_restore_default_clicked_deletes_voice_config_and_emits_signal(
    qapp: QApplication, tmp_path: Path
) -> None:
    pet_dir = _make_pet_dir(tmp_path)
    save_pet_voice_config(pet_dir, PetVoiceConfig(provider=TTSProviderName.EDGE, voice="x"))
    dialog = VoiceChangeDialog(pet_dir, tmp_path / "settings.json")
    updated: list[Path] = []
    dialog.voice_updated.connect(updated.append)

    dialog._on_restore_default_clicked()

    assert load_pet_voice_config(pet_dir) is None
    assert updated == [pet_dir]


def test_show_record_mode_shows_error_when_elevenlabs_key_missing(
    qapp: QApplication, tmp_path: Path
) -> None:
    pet_dir = _make_pet_dir(tmp_path)
    dialog = VoiceChangeDialog(pet_dir, tmp_path / "settings.json")

    dialog._show_record_mode()

    assert not dialog._mode_error_label.isHidden()
    assert "ElevenLabs API Key" in dialog._mode_error_label.text()
    assert dialog._stack.currentWidget() is dialog._mode_select_view
    assert dialog._reading_recording_widget is None


def test_on_recording_skipped_clears_widget_and_returns_to_mode_select(
    qapp: QApplication, tmp_path: Path
) -> None:
    pet_dir = _make_pet_dir(tmp_path)
    dialog = VoiceChangeDialog(pet_dir, tmp_path / "settings.json")
    dialog._reading_recording_widget = object()  # type: ignore[assignment]

    dialog._on_recording_skipped()

    assert dialog._reading_recording_widget is None
    assert dialog._stack.currentWidget() is dialog._mode_select_view


def test_show_cloning_page_starts_worker_when_key_present(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pet_dir = _make_pet_dir(tmp_path)
    dialog = VoiceChangeDialog(pet_dir, tmp_path / "settings.json")
    dialog._settings.voice_cloning = VoiceCloningConfig(elevenlabs_api_key="el-key")
    monkeypatch.setattr(VoiceCloneWorker, "start", lambda self: None)

    dialog._show_cloning_page(b"wav-bytes")

    assert dialog._voice_worker is not None
    assert dialog._voice_card is not None
    assert dialog._voice_card._status_label.text() == "声音克隆中…"


def test_on_voice_done_saves_config_and_emits_signal(qapp: QApplication, tmp_path: Path) -> None:
    pet_dir = _make_pet_dir(tmp_path)
    dialog = VoiceChangeDialog(pet_dir, tmp_path / "settings.json")
    updated: list[Path] = []
    dialog.voice_updated.connect(updated.append)

    dialog._on_voice_done("voice-123")

    assert dialog._voice_worker is None
    saved = load_pet_voice_config(pet_dir)
    assert saved is not None
    assert saved.provider == TTSProviderName.ELEVENLABS
    assert saved.voice == "voice-123"
    assert updated == [pet_dir]


def test_on_voice_failed_shows_card_failure(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pet_dir = _make_pet_dir(tmp_path)
    dialog = VoiceChangeDialog(pet_dir, tmp_path / "settings.json")
    dialog._settings.voice_cloning = VoiceCloningConfig(elevenlabs_api_key="el-key")
    monkeypatch.setattr(VoiceCloneWorker, "start", lambda self: None)
    dialog._show_cloning_page(b"wav-bytes")

    dialog._on_voice_failed("配额不足")

    assert dialog._voice_worker is None
    assert dialog._voice_card is not None
    assert dialog._voice_card._status_label.text() == "声音克隆失败：配额不足"


def test_on_manual_save_clicked_rejects_empty_voice(qapp: QApplication, tmp_path: Path) -> None:
    pet_dir = _make_pet_dir(tmp_path)
    dialog = VoiceChangeDialog(pet_dir, tmp_path / "settings.json")
    dialog._show_manual_mode()

    dialog._on_manual_save_clicked()

    assert not dialog._manual_error_label.isHidden()
    assert load_pet_voice_config(pet_dir) is None


def test_on_manual_save_clicked_saves_config_and_emits_signal(
    qapp: QApplication, tmp_path: Path
) -> None:
    pet_dir = _make_pet_dir(tmp_path)
    dialog = VoiceChangeDialog(pet_dir, tmp_path / "settings.json")
    dialog._show_manual_mode()
    dialog._manual_provider_combo.setCurrentText(TTSProviderName.OPENAI.value)
    dialog._manual_voice_edit.setText("alloy")
    dialog._manual_model_edit.setText("tts-1-hd")
    updated: list[Path] = []
    dialog.voice_updated.connect(updated.append)

    dialog._on_manual_save_clicked()

    saved = load_pet_voice_config(pet_dir)
    assert saved is not None
    assert saved.provider == TTSProviderName.OPENAI
    assert saved.voice == "alloy"
    assert saved.model == "tts-1-hd"
    assert updated == [pet_dir]


def test_build_manual_page_prefills_existing_voice_config(
    qapp: QApplication, tmp_path: Path
) -> None:
    pet_dir = _make_pet_dir(tmp_path)
    save_pet_voice_config(
        pet_dir, PetVoiceConfig(provider=TTSProviderName.ELEVENLABS, voice="voice-123")
    )
    dialog = VoiceChangeDialog(pet_dir, tmp_path / "settings.json")

    dialog._show_manual_mode()

    assert dialog._manual_provider_combo.currentText() == TTSProviderName.ELEVENLABS.value
    assert dialog._manual_voice_edit.text() == "voice-123"
