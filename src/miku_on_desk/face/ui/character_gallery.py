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
from miku_on_desk.face.ui.theme import (
    HOVER_COLOR,
    PRESSED_COLOR,
    SPACING_LG,
    SPACING_MD,
    TEAL_DARK,
    TEAL_MAIN,
    border_qss,
)

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
        self._is_current = is_current
        self._state = meta.fallback_state
        self._info = meta.states.get(self._state, meta.states[meta.fallback_state])
        self._elapsed_ms = 0

        layout = QVBoxLayout(self)
        self._sprite = PetSpriteWidget(
            meta, pet_dir / "spritesheet.png", scale=_TILE_SCALE, parent=self
        )
        layout.addWidget(self._sprite, alignment=Qt.AlignmentFlag.AlignHCenter)

        if is_current:
            badge = CaptionLabel("★ 当前角色", self)
            badge.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            layout.addWidget(badge)

        name_label = StrongBodyLabel(pet_dir.name, self)
        name_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(name_label)

        button = PrimaryPushButton("当前角色" if is_current else "切换到此角色", self)
        button.setEnabled(not is_current)
        button.clicked.connect(lambda: self.switch_requested.emit(self._pet_dir))
        layout.addWidget(button)

        self._idle_style = border_qss(TEAL_DARK) if is_current else ""
        self.setStyleSheet(f"CharacterStandTile {{ {self._idle_style} }}")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

    def _on_tick(self) -> None:
        self._elapsed_ms += _TICK_MS
        frame = frame_index(
            self._elapsed_ms / 1000,
            fps=self._info.fps,
            frame_count=self._info.frame_count,
            loop=True,
        )
        self._sprite.set_frame(self._state, frame)

    def showEvent(self, event: object) -> None:
        del event
        if not self._timer.isActive():
            self._timer.start(_TICK_MS)

    def hideEvent(self, event: object) -> None:
        """画廊面板整体隐藏/关闭时，Qt 会向所有可见子 widget 级联发出隐藏事件——借此
        暂停动画 tick，避免画廊关闭后展台仍在后台空转消耗 CPU。
        """
        del event
        self._timer.stop()

    def enterEvent(self, event: object) -> None:
        del event
        if not self._is_current:
            self.setStyleSheet(f"CharacterStandTile {{ {border_qss(HOVER_COLOR)} }}")

    def leaveEvent(self, event: object) -> None:
        del event
        self.setStyleSheet(f"CharacterStandTile {{ {self._idle_style} }}")


class _CreateCharacterTile(QWidget):
    """"＋ 创建新角色"格，虚线边框区分于普通角色展台。"""

    clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(160, 200)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._idle_style = border_qss(TEAL_MAIN, style="dashed")
        self._hover_style = border_qss(HOVER_COLOR, style="dashed")
        self._pressed_style = border_qss(PRESSED_COLOR, style="dashed")
        self.setStyleSheet(f"_CreateCharacterTile {{ {self._idle_style} }}")
        layout = QVBoxLayout(self)
        label = CaptionLabel("＋ 创建新角色", self)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

    def enterEvent(self, event: object) -> None:
        del event
        self.setStyleSheet(f"_CreateCharacterTile {{ {self._hover_style} }}")

    def leaveEvent(self, event: object) -> None:
        del event
        self.setStyleSheet(f"_CreateCharacterTile {{ {self._idle_style} }}")

    def mousePressEvent(self, event: object) -> None:
        del event
        self.setStyleSheet(f"_CreateCharacterTile {{ {self._pressed_style} }}")

    def mouseReleaseEvent(self, event: object) -> None:
        del event
        self.setStyleSheet(f"_CreateCharacterTile {{ {self._hover_style} }}")
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
        self._grid.setContentsMargins(SPACING_LG, SPACING_LG, SPACING_LG, SPACING_LG)
        self._grid.setSpacing(SPACING_MD)
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

        self._empty_label = CaptionLabel(
            "还没有角色，点击下方「＋」创建第一个角色", self._grid_container
        )
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row_offset = 0
        if not characters:
            self._grid.addWidget(
                self._empty_label,
                0,
                0,
                1,
                _COLUMNS,
                alignment=Qt.AlignmentFlag.AlignCenter,
            )
            row_offset = 1
        else:
            self._empty_label.hide()

        for index, (pet_dir, meta) in enumerate(characters):
            tile = CharacterStandTile(
                pet_dir,
                meta,
                is_current=(pet_dir == self._current_pet_dir),
                parent=self._grid_container,
            )
            tile.switch_requested.connect(self._on_switch_requested)
            self._grid.addWidget(tile, index // _COLUMNS + row_offset, index % _COLUMNS)

        create_tile = _CreateCharacterTile(self._grid_container)
        create_tile.clicked.connect(self.create_requested)
        create_index = len(characters)
        self._grid.addWidget(
            create_tile, create_index // _COLUMNS + row_offset, create_index % _COLUMNS
        )

    def _on_switch_requested(self, pet_dir: Path) -> None:
        self._current_pet_dir = pet_dir
        self.character_switched.emit(pet_dir)
        self._reload()

    def on_character_created(self, pet_dir: Path) -> None:
        """新角色生成完成后由调用方触发，切换当前选中项并重新扫描刷新网格。"""
        self._current_pet_dir = pet_dir
        self._reload()
