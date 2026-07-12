"""``KioskSettingsPanel`` 的角色切换/音量持久化/退出信号回归测试。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

from PIL import Image
from PySide6.QtWidgets import QApplication
from qfluentwidgets import PrimaryPushButton, PushButton

from miku_on_desk.config.settings import AppSettings
from miku_on_desk.face.ui.kiosk_settings_panel import (
    KioskSettingsPanel,
    _format_volume_percent,
    _parse_volume_percent,
)


def _make_character_dir(parent: Path, name: str) -> Path:
    pet_dir = parent / name
    pet_dir.mkdir()
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(pet_dir / "spritesheet.png")
    meta = {
        "pet_name": name,
        "frame_width": 4,
        "frame_height": 4,
        "columns": 1,
        "rows": 1,
        "fallback_state": "idle",
        "states": {"idle": {"row": 0, "frame_count": 1, "fps": 1.0, "loop": True}},
    }
    (pet_dir / "pet.json").write_text(json.dumps(meta), encoding="utf-8")
    return pet_dir


def test_parse_volume_percent_round_trips_with_format() -> None:
    assert _parse_volume_percent(_format_volume_percent(-30)) == -30
    assert _parse_volume_percent(_format_volume_percent(0)) == 0
    assert _parse_volume_percent(_format_volume_percent(50)) == 50


def test_parse_volume_percent_falls_back_to_zero_on_garbage() -> None:
    assert _parse_volume_percent("not a percent") == 0


def test_current_pet_button_is_disabled_and_others_enabled(
    qapp: QApplication, tmp_path: Path
) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    current = _make_character_dir(assets_dir, "current_pet")
    other = _make_character_dir(assets_dir, "other_pet")
    settings_path = tmp_path / "settings.json"
    AppSettings().save(settings_path)

    panel = KioskSettingsPanel(assets_dir, current, settings_path)

    assert panel._character_buttons[current].isEnabled() is False
    assert panel._character_buttons[other].isEnabled() is True


def test_clicking_other_character_button_emits_character_switched(
    qapp: QApplication, tmp_path: Path
) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    current = _make_character_dir(assets_dir, "current_pet")
    other = _make_character_dir(assets_dir, "other_pet")
    settings_path = tmp_path / "settings.json"
    AppSettings().save(settings_path)

    panel = KioskSettingsPanel(assets_dir, current, settings_path)
    on_switched = Mock()
    panel.character_switched.connect(on_switched)

    panel._character_buttons[other].click()

    on_switched.assert_called_once_with(other)
    assert panel._character_buttons[current].isEnabled() is True
    assert panel._character_buttons[other].isEnabled() is False


def test_volume_slider_initializes_from_settings_and_persists_changes(
    qapp: QApplication, tmp_path: Path
) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    current = _make_character_dir(assets_dir, "current_pet")
    settings_path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.tts.volume = "-20%"
    settings.save(settings_path)

    panel = KioskSettingsPanel(assets_dir, current, settings_path)
    assert panel._volume_slider.value() == -20

    panel._volume_slider.setValue(15)

    reloaded = AppSettings.load(settings_path)
    assert reloaded.tts.volume == "+15%"


def test_quit_button_emits_quit_requested(qapp: QApplication, tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    current = _make_character_dir(assets_dir, "current_pet")
    settings_path = tmp_path / "settings.json"
    AppSettings().save(settings_path)

    panel = KioskSettingsPanel(assets_dir, current, settings_path)
    on_quit = Mock()
    panel.quit_requested.connect(on_quit)

    quit_buttons = [
        button
        for button in panel.findChildren(PrimaryPushButton)
        if button.text() == "退出应用"
    ]
    assert len(quit_buttons) == 1
    quit_buttons[0].click()

    on_quit.assert_called_once()


def test_clone_button_emits_clone_requested(qapp: QApplication, tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    current = _make_character_dir(assets_dir, "current_pet")
    settings_path = tmp_path / "settings.json"
    AppSettings().save(settings_path)

    panel = KioskSettingsPanel(assets_dir, current, settings_path)
    on_clone = Mock()
    panel.clone_requested.connect(on_clone)

    clone_buttons = [
        button for button in panel.findChildren(PushButton) if button.text() == "＋ 克隆新角色"
    ]
    assert len(clone_buttons) == 1
    clone_buttons[0].click()

    on_clone.assert_called_once()
