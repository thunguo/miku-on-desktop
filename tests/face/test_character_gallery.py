"""``CharacterGalleryPanel`` 的角色扫描/切换/创建信号联动回归测试。"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from PySide6.QtWidgets import QApplication
from qfluentwidgets import PrimaryPushButton

from miku_on_desk.face.sprite_sheet import SpriteSheetMeta
from miku_on_desk.face.ui.character_gallery import (
    CharacterGalleryPanel,
    CharacterStandTile,
    _CreateCharacterTile,
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


def test_reload_renders_character_tiles_plus_create_tile(
    qapp: QApplication, tmp_path: Path
) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    pet_a = _make_character_dir(assets_dir, "pet_a")
    _make_character_dir(assets_dir, "pet_b")
    panel = CharacterGalleryPanel(assets_dir, pet_a)

    assert panel._grid.count() == 3


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


def test_on_character_created_sets_current_and_reloads(
    qapp: QApplication, tmp_path: Path
) -> None:
    assets_dir = tmp_path / "assets_pets"
    assets_dir.mkdir()
    pet_a = _make_character_dir(assets_dir, "pet_a")
    panel = CharacterGalleryPanel(assets_dir, pet_a)
    pet_b = _make_character_dir(assets_dir, "pet_b")

    panel.on_character_created(pet_b)

    assert panel._current_pet_dir == pet_b
    assert panel._grid.count() == 3


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
