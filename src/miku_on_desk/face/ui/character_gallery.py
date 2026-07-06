"""角色画廊面板：展示 ``assets/pets/`` 下动态发现的所有角色，一格一个动画展台，支持切换到
任意角色或创建新角色。

用普通 ``QWidget`` + ``QGridLayout``（套 ``QScrollArea``）而不是 ``FluentWindow`` 多标签页
——画廊是单页可视网格，不是配置表单。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QGridLayout, QScrollArea, QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, PrimaryPushButton, StrongBodyLabel

from miku_on_desk.face.sprite_sheet import SpriteSheetMeta, SpriteSheetMetaError, frame_index
from miku_on_desk.face.ui.sprite_widget import PetSpriteWidget
from miku_on_desk.face.ui.theme import TEAL_DARK, TEAL_MAIN

_TICK_MS = 33
_TILE_SCALE = 1.0
_COLUMNS = 4


class CharacterStandTile(QWidget):
    """单个角色展台：循环播放该角色的 idle（或 fallback）动画 + 名称 + 切换按钮。"""

    switch_requested = Signal(Path)

    def __init__(
        self,
        pet_dir: Path,
        meta: SpriteSheetMeta,
        *,
        is_current: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._pet_dir = pet_dir
        self._state = meta.fallback_state
        self._info = meta.states.get(self._state, meta.states[meta.fallback_state])
        self._elapsed_ms = 0

        layout = QVBoxLayout(self)
        self._sprite = PetSpriteWidget(
            meta, pet_dir / "spritesheet.png", scale=_TILE_SCALE, parent=self
        )
        layout.addWidget(self._sprite, alignment=Qt.AlignmentFlag.AlignHCenter)

        name_label = StrongBodyLabel(pet_dir.name, self)
        name_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(name_label)

        button = PrimaryPushButton("当前角色" if is_current else "切换到此角色", self)
        button.setEnabled(not is_current)
        button.clicked.connect(lambda: self.switch_requested.emit(self._pet_dir))
        layout.addWidget(button)

        if is_current:
            self.setStyleSheet(
                f"CharacterStandTile {{ border: 2px solid {TEAL_DARK}; border-radius: 8px; }}"
            )

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(_TICK_MS)

    def _on_tick(self) -> None:
        self._elapsed_ms += _TICK_MS
        frame = frame_index(
            self._elapsed_ms / 1000,
            fps=self._info.fps,
            frame_count=self._info.frame_count,
            loop=True,
        )
        self._sprite.set_frame(self._state, frame)


class _CreateCharacterTile(QWidget):
    """"＋ 创建新角色"格，虚线边框区分于普通角色展台。"""

    clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(160, 200)
        self.setStyleSheet(
            f"_CreateCharacterTile {{ border: 2px dashed {TEAL_MAIN}; border-radius: 8px; }}"
        )
        layout = QVBoxLayout(self)
        label = CaptionLabel("＋ 创建新角色", self)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

    def mouseReleaseEvent(self, event: object) -> None:
        del event
        self.clicked.emit()


class CharacterGalleryPanel(QWidget):
    """扫描 ``assets_pets_dir`` 下的角色目录，渲染展台网格 + "创建新角色"格。"""

    character_switched = Signal(Path)
    create_requested = Signal()

    def __init__(
        self,
        assets_pets_dir: Path,
        current_pet_dir: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._assets_pets_dir = assets_pets_dir
        self._current_pet_dir = current_pet_dir
        self.resize(720, 560)

        outer = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        self._grid_container = QWidget(scroll)
        self._grid = QGridLayout(self._grid_container)
        scroll.setWidget(self._grid_container)

        self._reload()

    def _scan_characters(self) -> list[tuple[Path, SpriteSheetMeta]]:
        found: list[tuple[Path, SpriteSheetMeta]] = []
        if not self._assets_pets_dir.is_dir():
            return found
        for child in sorted(self._assets_pets_dir.iterdir()):
            if not child.is_dir():
                continue
            try:
                meta = SpriteSheetMeta.load(child / "pet.json")
            except (SpriteSheetMetaError, FileNotFoundError):
                continue
            found.append((child, meta))
        return found

    def _reload(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()

        characters = self._scan_characters()
        for index, (pet_dir, meta) in enumerate(characters):
            tile = CharacterStandTile(
                pet_dir,
                meta,
                is_current=(pet_dir == self._current_pet_dir),
                parent=self._grid_container,
            )
            tile.switch_requested.connect(self._on_switch_requested)
            self._grid.addWidget(tile, index // _COLUMNS, index % _COLUMNS)

        create_tile = _CreateCharacterTile(self._grid_container)
        create_tile.clicked.connect(self.create_requested)
        create_index = len(characters)
        self._grid.addWidget(create_tile, create_index // _COLUMNS, create_index % _COLUMNS)

    def _on_switch_requested(self, pet_dir: Path) -> None:
        self._current_pet_dir = pet_dir
        self.character_switched.emit(pet_dir)
        self._reload()

    def on_character_created(self, pet_dir: Path) -> None:
        """新角色生成完成后由调用方触发，切换当前选中项并重新扫描刷新网格。"""
        self._current_pet_dir = pet_dir
        self._reload()
