"""桌宠精灵图元数据（`pet.json`）解析与帧号/裁剪矩形的纯函数。

网格布局：每个状态占一整行，`frame_width`/`frame_height`/`columns`/`rows` 均在
`pet.json` 里声明而非硬编码；`fallback_state` 让代码里未来新增的 `PetState`
在旧资产缺对应行时优雅降级，而不是直接抛异常。
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, Field, ValidationError

from miku_on_desk.face.pet_state import PetState


class SpriteSheetMetaError(Exception):
    """`pet.json` 内容未通过解析或校验。"""


class Rect(NamedTuple):
    x: int
    y: int
    width: int
    height: int


class StateSpriteInfo(BaseModel):
    row: int
    frame_count: int
    fps: float
    loop: bool


class SpriteSheetMeta(BaseModel):
    pet_name: str
    frame_width: int
    frame_height: int
    columns: int
    rows: int
    fallback_state: PetState = PetState.IDLE
    states: dict[PetState, StateSpriteInfo] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> SpriteSheetMeta:
        try:
            meta = cls.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise SpriteSheetMetaError(f"解析 {path} 失败：{exc}") from exc

        if meta.fallback_state not in meta.states:
            raise SpriteSheetMetaError(
                f"{path} 的 fallback_state {meta.fallback_state!r} 未出现在 states 中"
            )
        for state, info in meta.states.items():
            if not 0 <= info.row < meta.rows:
                raise SpriteSheetMetaError(
                    f"{path} 中状态 {state!r} 的 row={info.row} 超出 rows={meta.rows} 范围"
                )
            if not 0 < info.frame_count <= meta.columns:
                raise SpriteSheetMetaError(
                    f"{path} 中状态 {state!r} 的 frame_count={info.frame_count} "
                    f"超出 columns={meta.columns} 范围"
                )
            if info.fps <= 0:
                raise SpriteSheetMetaError(f"{path} 中状态 {state!r} 的 fps 必须为正数")
        return meta


def frame_index(elapsed_in_state: float, *, fps: float, frame_count: int, loop: bool) -> int:
    """给定在当前状态内已播放的时长,算出应显示第几帧。

    ``loop=True``：循环播放，超出总帧数后回绕（``% frame_count``）。
    ``loop=False``：播完后定格在最后一帧，不回绕。
    """
    raw_index = int(elapsed_in_state * fps)
    if loop:
        return raw_index % frame_count
    return min(raw_index, frame_count - 1)


def cell_rect(meta: SpriteSheetMeta, state: PetState, frame: int) -> Rect:
    """算出 ``state``/``frame`` 对应的裁剪矩形；``state`` 不在 ``meta.states`` 时
    退回 ``meta.fallback_state``，而非抛异常（资产缺行时的优雅降级）。
    """
    info = meta.states.get(state, meta.states[meta.fallback_state])
    return Rect(
        x=frame * meta.frame_width,
        y=info.row * meta.frame_height,
        width=meta.frame_width,
        height=meta.frame_height,
    )
