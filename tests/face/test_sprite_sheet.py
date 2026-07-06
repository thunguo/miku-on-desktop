"""SpriteSheetMeta 解析/校验与 frame_index/cell_rect 纯函数的回归测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from miku_on_desk.face.pet_state import PetState
from miku_on_desk.face.sprite_sheet import (
    Rect,
    SpriteSheetMeta,
    SpriteSheetMetaError,
    cell_rect,
    frame_index,
)

_VALID_META: dict = {
    "pet_name": "miku_pixel",
    "frame_width": 128,
    "frame_height": 128,
    "columns": 8,
    "rows": 2,
    "fallback_state": "idle",
    "states": {
        "idle": {"row": 0, "frame_count": 6, "fps": 6.0, "loop": True},
        "talking": {"row": 1, "frame_count": 8, "fps": 10.0, "loop": True},
    },
}


def _write_meta(path: Path, data: dict) -> Path:
    meta_path = path / "pet.json"
    meta_path.write_text(json.dumps(data), encoding="utf-8")
    return meta_path


def test_load_valid_meta_parses_states_as_pet_state_keys(tmp_path: Path) -> None:
    meta_path = _write_meta(tmp_path, _VALID_META)
    meta = SpriteSheetMeta.load(meta_path)

    assert meta.pet_name == "miku_pixel"
    assert meta.states[PetState.IDLE].frame_count == 6
    assert meta.states[PetState.TALKING].fps == 10.0


def test_load_rejects_malformed_json(tmp_path: Path) -> None:
    meta_path = tmp_path / "pet.json"
    meta_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(SpriteSheetMetaError):
        SpriteSheetMeta.load(meta_path)


def test_load_rejects_fallback_state_missing_from_states(tmp_path: Path) -> None:
    data = dict(_VALID_META)
    data["fallback_state"] = "error"
    meta_path = _write_meta(tmp_path, data)

    with pytest.raises(SpriteSheetMetaError, match="fallback_state"):
        SpriteSheetMeta.load(meta_path)


def test_load_rejects_row_out_of_bounds(tmp_path: Path) -> None:
    data = json.loads(json.dumps(_VALID_META))
    data["states"]["talking"]["row"] = 99
    meta_path = _write_meta(tmp_path, data)

    with pytest.raises(SpriteSheetMetaError, match="row"):
        SpriteSheetMeta.load(meta_path)


def test_load_rejects_frame_count_exceeding_columns(tmp_path: Path) -> None:
    data = json.loads(json.dumps(_VALID_META))
    data["states"]["talking"]["frame_count"] = 99
    meta_path = _write_meta(tmp_path, data)

    with pytest.raises(SpriteSheetMetaError, match="frame_count"):
        SpriteSheetMeta.load(meta_path)


def test_load_rejects_non_positive_fps(tmp_path: Path) -> None:
    data = json.loads(json.dumps(_VALID_META))
    data["states"]["talking"]["fps"] = 0.0
    meta_path = _write_meta(tmp_path, data)

    with pytest.raises(SpriteSheetMetaError, match="fps"):
        SpriteSheetMeta.load(meta_path)


def test_frame_index_loops_when_loop_is_true() -> None:
    assert frame_index(0.0, fps=10.0, frame_count=4, loop=True) == 0
    assert frame_index(0.35, fps=10.0, frame_count=4, loop=True) == 3
    assert frame_index(1.0, fps=10.0, frame_count=4, loop=True) == 2


def test_frame_index_holds_last_frame_when_loop_is_false() -> None:
    assert frame_index(0.0, fps=10.0, frame_count=4, loop=False) == 0
    assert frame_index(0.35, fps=10.0, frame_count=4, loop=False) == 3
    assert frame_index(10.0, fps=10.0, frame_count=4, loop=False) == 3


def test_cell_rect_uses_frame_and_row_offsets(tmp_path: Path) -> None:
    meta_path = _write_meta(tmp_path, _VALID_META)
    meta = SpriteSheetMeta.load(meta_path)

    rect = cell_rect(meta, PetState.TALKING, 2)

    assert rect == Rect(x=256, y=128, width=128, height=128)


def test_cell_rect_falls_back_to_fallback_state_for_unknown_state(tmp_path: Path) -> None:
    meta_path = _write_meta(tmp_path, _VALID_META)
    meta = SpriteSheetMeta.load(meta_path)

    rect = cell_rect(meta, PetState.ERROR, 0)

    assert rect == Rect(x=0, y=0, width=128, height=128)
