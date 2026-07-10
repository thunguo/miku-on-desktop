"""``CharacterGalleryPanel`` 的角色扫描/切换/创建信号联动回归测试。"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from PySide6.QtWidgets import QApplication
from qfluentwidgets import CaptionLabel, PrimaryPushButton, PushButton

from miku_on_desk.config.settings import TTSProviderName
from miku_on_desk.face.character_voice import PetVoiceConfig, save_pet_voice_config
from miku_on_desk.face.sprite_sheet import SpriteSheetMeta
from miku_on_desk.face.ui.character_gallery import (
    CharacterGalleryPanel,
    CharacterStandTile,
    _CloneCharacterTile,
    _CreateCharacterTile,
)
from miku_on_desk.face.ui.theme import SPACING_LG, SPACING_MD


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


def test_scan_characters_returns_empty_list_when_assets_dir_missing(
    qapp: QApplication, tmp_path: Path
) -> None:
    panel = CharacterGalleryPanel(tmp_path / "does_not_exist", tmp_path / "current")

    assert panel._scan_characters() == []


def test_scan_characters_skips_dirs_without_pet_json(qapp: QApplication, tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    valid = _make_character_dir(assets_dir, "valid_pet")
    (assets_dir / "no_json_pet").mkdir()
    panel = CharacterGalleryPanel(assets_dir, valid)

    found = panel._scan_characters()

    assert [pet_dir for pet_dir, _meta in found] == [valid]


def test_scan_characters_skips_dirs_with_invalid_pet_json(
    qapp: QApplication, tmp_path: Path
) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    valid = _make_character_dir(assets_dir, "valid_pet")
    broken = assets_dir / "broken_pet"
    broken.mkdir()
    (broken / "pet.json").write_text("not valid json", encoding="utf-8")
    panel = CharacterGalleryPanel(assets_dir, valid)

    found = panel._scan_characters()

    assert [pet_dir for pet_dir, _meta in found] == [valid]


def test_reload_renders_character_tiles_plus_create_and_clone_tiles(
    qapp: QApplication, tmp_path: Path
) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    pet_a = _make_character_dir(assets_dir, "pet_a")
    _make_character_dir(assets_dir, "pet_b")
    panel = CharacterGalleryPanel(assets_dir, pet_a)

    assert panel._grid.count() == 4


def test_on_switch_requested_updates_current_and_emits_character_switched(
    qapp: QApplication, tmp_path: Path
) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    pet_a = _make_character_dir(assets_dir, "pet_a")
    pet_b = _make_character_dir(assets_dir, "pet_b")
    panel = CharacterGalleryPanel(assets_dir, pet_a)
    switched: list[Path] = []
    panel.character_switched.connect(switched.append)

    panel._on_switch_requested(pet_b)

    assert switched == [pet_b]
    assert panel._current_pet_dir == pet_b


def test_on_character_created_sets_current_and_reloads(qapp: QApplication, tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    pet_a = _make_character_dir(assets_dir, "pet_a")
    panel = CharacterGalleryPanel(assets_dir, pet_a)
    pet_b = _make_character_dir(assets_dir, "pet_b")

    panel.on_character_created(pet_b)

    assert panel._current_pet_dir == pet_b
    assert panel._grid.count() == 4


def test_character_stand_tile_button_disabled_and_labelled_when_current(
    qapp: QApplication, tmp_path: Path
) -> None:
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")
    tile = CharacterStandTile(pet_dir, meta, is_current=True)

    button = tile.findChild(PrimaryPushButton)
    assert button is not None
    assert button.isEnabled() is False
    assert button.text() == "当前角色"


def test_character_stand_tile_button_emits_switch_requested_when_clicked(
    qapp: QApplication, tmp_path: Path
) -> None:
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")
    tile = CharacterStandTile(pet_dir, meta, is_current=False)
    switched: list[Path] = []
    tile.switch_requested.connect(switched.append)

    button = tile.findChild(PrimaryPushButton)
    assert button is not None
    assert button.text() == "切换到此角色"
    button.click()

    assert switched == [pet_dir]


def test_create_character_tile_emits_clicked_on_mouse_release(qapp: QApplication) -> None:
    tile = _CreateCharacterTile()
    clicks: list[None] = []
    tile.clicked.connect(lambda: clicks.append(None))

    tile.mouseReleaseEvent(None)

    assert clicks == [None]


def test_clone_character_tile_emits_clicked_on_mouse_release(qapp: QApplication) -> None:
    tile = _CloneCharacterTile()
    clicks: list[None] = []
    tile.clicked.connect(lambda: clicks.append(None))

    tile.mouseReleaseEvent(None)

    assert clicks == [None]


def test_clone_tile_click_emits_panel_clone_requested(qapp: QApplication, tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    pet_a = _make_character_dir(assets_dir, "pet_a")
    panel = CharacterGalleryPanel(assets_dir, pet_a)
    requested: list[None] = []
    panel.clone_requested.connect(lambda: requested.append(None))
    clone_tile = panel.findChild(_CloneCharacterTile)
    assert clone_tile is not None

    clone_tile.mouseReleaseEvent(None)

    assert requested == [None]


def test_character_stand_tile_voice_button_emits_voice_change_requested_when_clicked(
    qapp: QApplication, tmp_path: Path
) -> None:
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")
    tile = CharacterStandTile(pet_dir, meta, is_current=False)
    requested: list[Path] = []
    tile.voice_change_requested.connect(requested.append)

    voice_buttons = [
        button for button in tile.findChildren(PushButton) if button.text() == "更换声音"
    ]
    assert len(voice_buttons) == 1
    voice_buttons[0].click()

    assert requested == [pet_dir]


def test_voice_change_requested_from_tile_propagates_through_panel(
    qapp: QApplication, tmp_path: Path
) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    pet_a = _make_character_dir(assets_dir, "pet_a")
    panel = CharacterGalleryPanel(assets_dir, pet_a)
    requested: list[Path] = []
    panel.voice_change_requested.connect(requested.append)
    tile = panel.findChild(CharacterStandTile)
    assert tile is not None

    voice_buttons = [
        button for button in tile.findChildren(PushButton) if button.text() == "更换声音"
    ]
    assert len(voice_buttons) == 1
    voice_buttons[0].click()

    assert requested == [pet_a]


def test_character_stand_tile_hides_voice_badge_when_no_voice_bound(
    qapp: QApplication, tmp_path: Path
) -> None:
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")

    tile = CharacterStandTile(pet_dir, meta, is_current=False)

    assert not any("已绑定专属声音" in child.text() for child in tile.findChildren(CaptionLabel))


def test_character_stand_tile_shows_voice_badge_when_voice_bound(
    qapp: QApplication, tmp_path: Path
) -> None:
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    save_pet_voice_config(
        pet_dir, PetVoiceConfig(provider=TTSProviderName.ELEVENLABS, voice="voice-123")
    )
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")

    tile = CharacterStandTile(pet_dir, meta, is_current=False)

    assert any("已绑定专属声音" in child.text() for child in tile.findChildren(CaptionLabel))


def test_reload_shows_empty_state_message_when_no_characters(
    qapp: QApplication, tmp_path: Path
) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    panel = CharacterGalleryPanel(assets_dir, tmp_path / "current")
    panel.show()

    assert panel._empty_label.isVisibleTo(panel) is True
    assert panel._grid.count() == 3


def test_reload_hides_empty_state_message_when_characters_exist(
    qapp: QApplication, tmp_path: Path
) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    pet_a = _make_character_dir(assets_dir, "pet_a")
    panel = CharacterGalleryPanel(assets_dir, pet_a)
    panel.show()

    assert panel._empty_label.isVisibleTo(panel) is False


def test_grid_has_configured_margins_and_spacing(qapp: QApplication, tmp_path: Path) -> None:
    panel = CharacterGalleryPanel(tmp_path / "does_not_exist", tmp_path / "current")

    assert panel._grid.spacing() == SPACING_MD
    assert panel._grid.getContentsMargins() == (SPACING_LG, SPACING_LG, SPACING_LG, SPACING_LG)


# ── 可见性驱动的 tick 暂停（阶段 F） ────────────────────────────────────────


def test_character_stand_tile_timer_not_running_before_shown(
    qapp: QApplication, tmp_path: Path
) -> None:
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")
    tile = CharacterStandTile(pet_dir, meta, is_current=False)

    assert tile._timer.isActive() is False


def test_character_stand_tile_show_starts_timer_and_hide_stops_it(
    qapp: QApplication, tmp_path: Path
) -> None:
    pet_dir = _make_character_dir(tmp_path, "pet_a")
    meta = SpriteSheetMeta.load(pet_dir / "pet.json")
    tile = CharacterStandTile(pet_dir, meta, is_current=False)

    tile.show()
    assert tile._timer.isActive() is True

    tile.hide()
    assert tile._timer.isActive() is False


def test_gallery_panel_hide_stops_all_tile_timers(qapp: QApplication, tmp_path: Path) -> None:
    """画廊面板整体关闭/隐藏时，Qt 会向所有可见子 widget 级联发出隐藏事件——展台的
    动画 tick 应该跟着暂停，避免面板关闭后仍在后台空转消耗 CPU。
    """
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    pet_a = _make_character_dir(assets_dir, "pet_a")
    panel = CharacterGalleryPanel(assets_dir, pet_a)
    panel.show()
    tile = panel.findChild(CharacterStandTile)
    assert tile is not None
    assert tile._timer.isActive() is True

    panel.hide()

    assert tile._timer.isActive() is False
