"""树莓派 kiosk 硬件端的精简触屏设置面板：只做切角色、语音音量、退出应用、关于——
不是 ``settings_panel.py::SettingsPanel``（680x560 FluentWindow，Providers/Persona/
Permissions/Skills/Memory 等全量配置）的裁剪版，是专为触屏小屏幕设计的轻量弹层。复杂
配置项完全不在本机呈现，改由 ``kiosk_main.py`` 额外启动的局域网 Web 管理页面
（``web/settings_server.py``）编辑。

"角色克隆"向导入口（拍照生成外观 + 录音克隆声音）依赖树莓派本机的摄像头/麦克风，必须
留在本机——浏览器远程访问 Web 页面时用的是访问者自己设备的摄像头/麦克风，不是树莓派
本机的硬件。这里只发出 ``clone_requested`` 信号，具体打开哪个对话框由调用方
（``kiosk_main.py``，与 ``main.py::_open_character_gallery`` 接线 ``CharacterCloneDialog``
是同一个模式）决定，本模块不直接依赖 ``character_clone_dialog.py``。
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, PrimaryPushButton, PushButton, Slider, StrongBodyLabel

from miku_on_desk.config.settings import AppSettings
from miku_on_desk.face.ui.character_gallery import discover_pet_dirs

_VOLUME_MIN_PERCENT = -50
_VOLUME_MAX_PERCENT = 50


def _app_version() -> str:
    try:
        return version("miku-on-desk")
    except PackageNotFoundError:
        return "unknown"


def _parse_volume_percent(value: str) -> int:
    try:
        return int(value.strip().removesuffix("%"))
    except ValueError:
        return 0


def _format_volume_percent(value: int) -> str:
    return f"{value:+d}%"


class KioskSettingsPanel(QWidget):
    character_switched = Signal(Path)
    clone_requested = Signal()
    quit_requested = Signal()

    def __init__(
        self,
        assets_pets_dir: Path,
        current_pet_dir: Path,
        settings_path: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._assets_pets_dir = assets_pets_dir
        self._current_pet_dir = current_pet_dir
        self._settings_path = settings_path
        self.resize(300, 440)

        layout = QVBoxLayout(self)

        close_button = PushButton("×", self)
        close_button.setFixedWidth(40)
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addWidget(StrongBodyLabel("切换角色", self))
        self._character_buttons: dict[Path, PrimaryPushButton] = {}
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        character_container = QWidget(scroll)
        character_layout = QVBoxLayout(character_container)
        for pet_dir, _meta in discover_pet_dirs(assets_pets_dir):
            button = PrimaryPushButton(pet_dir.name, character_container)
            button.setEnabled(pet_dir != current_pet_dir)
            button.clicked.connect(lambda _checked=False, d=pet_dir: self._on_switch(d))
            self._character_buttons[pet_dir] = button
            character_layout.addWidget(button)
        clone_button = PushButton("＋ 克隆新角色", character_container)
        clone_button.clicked.connect(self.clone_requested)
        character_layout.addWidget(clone_button)
        scroll.setWidget(character_container)
        layout.addWidget(scroll)

        layout.addWidget(StrongBodyLabel("语音音量", self))
        settings = AppSettings.load(settings_path)
        self._volume_slider = Slider(self)
        self._volume_slider.setOrientation(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(_VOLUME_MIN_PERCENT, _VOLUME_MAX_PERCENT)
        self._volume_slider.setValue(_parse_volume_percent(settings.tts.volume))
        self._volume_slider.valueChanged.connect(self._on_volume_changed)
        layout.addWidget(self._volume_slider)

        layout.addStretch()

        quit_button = PrimaryPushButton("退出应用", self)
        quit_button.clicked.connect(self.quit_requested)
        layout.addWidget(quit_button)

        layout.addWidget(CaptionLabel(f"miku-on-desk kiosk · {_app_version()}", self))

    def _on_switch(self, pet_dir: Path) -> None:
        if pet_dir == self._current_pet_dir:
            return
        previous_button = self._character_buttons.get(self._current_pet_dir)
        if previous_button is not None:
            previous_button.setEnabled(True)
        self._character_buttons[pet_dir].setEnabled(False)
        self._current_pet_dir = pet_dir
        self.character_switched.emit(pet_dir)

    def _on_volume_changed(self, value: int) -> None:
        settings = AppSettings.load(self._settings_path)
        settings.tts.volume = _format_volume_percent(value)
        settings.save(self._settings_path)
