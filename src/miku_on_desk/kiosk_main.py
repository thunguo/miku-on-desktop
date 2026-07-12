"""树莓派硬件端 kiosk 入口：全屏铺满、默认展示 ``settings.kiosk.default_pet`` 指定的
角色、右上角设置图标打开精简触屏面板，另起一个局域网 Web 管理页面承载复杂配置编辑。
见 ``pyproject.toml`` 的 ``miku-on-desk-kiosk`` console script。

跟桌面入口 ``main.py`` 平级、彼此不互相依赖：``main.py`` 保持完全不变，这里只复用它
提取出的共享 setup 函数（``load_app_config``/``start_brain_runtime``）和几个纯 wiring
helper（``_assets_pets_dir``/``_on_character_switched``/``_open_character_gallery`` 等），
不重新实现这部分组装逻辑。两个入口靠"启动的是哪个命令"区分，不靠配置里的开关字段。

跟桌面版的几处刻意差异（都是"这个东西在没有桌面 shell、没有物理键盘鼠标的全屏一体机上
没有意义"，不是遗漏）：
- 不启动 hook server / 不安装任何 CLI hook——那是给"本机安装的编码 agent"用的旁路，
  树莓派 kiosk 上不会有人在这上面跑 Claude Code。
- 不建 ``GlobalHotKeyManager``——没有物理键盘可触发全局热键；触摸屏改用点一下精灵本体
  打开聊天弹窗（见 ``overlay_window.py::mouseReleaseEvent`` 的 kiosk 分支）。
- 不建系统托盘图标——kiosk 没有桌面面板可以放托盘。
- 不起访客路过定时器（``_start_visitor_scheduler``）——那是给"贴在桌面任意位置的悬浮窗
  旁边再弹一个"设计的效果，全屏单角色场景不适用。
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from qfluentwidgets import MessageBox

from miku_on_desk.brain.proactive import ProactiveToggleRequest
from miku_on_desk.brain.secrets.vault import SecretVault, default_vault_paths
from miku_on_desk.config import AppSettings
from miku_on_desk.face.hooks.session_report import GrowthStore
from miku_on_desk.face.relationship_store import RelationshipStore
from miku_on_desk.face.ui.character_clone_dialog import CharacterCloneDialog
from miku_on_desk.face.ui.kiosk_settings_panel import KioskSettingsPanel
from miku_on_desk.face.ui.overlay_window import OverlayWindow
from miku_on_desk.face.ui.rotated_container import RotatedContainer
from miku_on_desk.face.ui.settings_panel import SettingsPanel
from miku_on_desk.face.ui.speech_controller import SpeechController
from miku_on_desk.face.ui.theme import apply_fluent_theme
from miku_on_desk.main import (
    _SHUTDOWN,
    PetActions,
    _assets_pets_dir,
    _build_speech_controller,
    _build_voice_input,
    _on_character_switched,
    _open_character_gallery,
    _open_memory_panel,
    _open_recollection_gallery,
    _open_settings_panel,
    _startup_health_warnings,
    load_app_config,
    start_brain_runtime,
)
from miku_on_desk.web.settings_server import SettingsServer

logger = logging.getLogger(__name__)

_STARTUP_RETRY_EXIT_CODE = 75


def _startup_failure_message(error_id: str) -> str:
    """返回无需 Linux 桌面也能读懂的启动失败说明。

    systemd 会把完整 traceback 写进 journal；屏幕上只显示短错误编号，避免把凭证、路径或
    Python 内部细节暴露到常亮的实体设备上。
    """
    return (
        "Miku 启动失败\n\n"
        f"错误编号：{error_id}\n"
        "请通过 SSH 执行 journalctl -u miku-kiosk.service 查看详细日志。\n"
        "修复后点击“重试”。"
    )


def _show_startup_failure(error: Exception) -> int:
    """在 Qt 已可初始化时保留全屏错误页，而不是回落到 Linux 桌面。"""
    error_id = f"KIOSK-{uuid.uuid4().hex[:8].upper()}"
    logger.exception("Kiosk 启动失败（%s）：%s", error_id, error)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    assert isinstance(app, QApplication)
    app.setQuitOnLastWindowClosed(False)

    panel = QWidget()
    panel.setWindowTitle("Miku 启动失败")
    layout = QVBoxLayout(panel)
    message = QLabel(_startup_failure_message(error_id), panel)
    message.setWordWrap(True)
    retry = QPushButton("重试", panel)
    retry.clicked.connect(lambda: app.exit(_STARTUP_RETRY_EXIT_CODE))
    layout.addStretch()
    layout.addWidget(message)
    layout.addWidget(retry)
    layout.addStretch()
    panel.showFullScreen()
    return app.exec()


def _kiosk_default_pet_dir(settings: AppSettings) -> Path:
    return _assets_pets_dir() / settings.kiosk.default_pet


def _open_kiosk_clone_dialog(
    window: OverlayWindow,
    settings_path: Path,
    open_windows: list[QWidget],
    *,
    vault: SecretVault,
    speech_controller: SpeechController | None,
    relationship_store: RelationshipStore,
) -> CharacterCloneDialog:
    """跟 ``main.py::_open_character_clone_dialog`` 同样的接线，但不依赖
    ``CharacterGalleryPanel``——kiosk 用的是 ``KioskSettingsPanel``，克隆完成后不刷新
    面板的角色按钮列表（下次重新打开面板会读到最新角色），只负责真正切换到新角色。
    """
    dialog = CharacterCloneDialog(_assets_pets_dir(), settings_path, vault=vault)
    dialog.character_created.connect(
        lambda pet_dir: _on_character_switched(
            pet_dir,
            window,
            settings_path,
            speech_controller=speech_controller,
            vault=vault,
            relationship_store=relationship_store,
        )
    )
    open_windows.append(dialog)
    dialog.show()
    return dialog


def _run_kiosk() -> None:
    config = load_app_config()
    bootstrap = config.bootstrap
    settings_path = config.settings_path
    vault = config.vault
    settings = config.settings

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    apply_fluent_theme()

    runtime = start_brain_runtime(config)
    event_bus = runtime.event_bus
    confirm_gate = runtime.confirm_gate
    cancellation_gate = runtime.cancellation_gate
    message_queue = runtime.message_queue
    chat_input = runtime.chat_input
    memory_system = runtime.memory_system
    brain_thread = runtime.brain_thread

    growth_store = GrowthStore(bootstrap.resolve_data_dir() / "companion_growth.json")
    relationship_store = RelationshipStore(
        bootstrap.resolve_data_dir() / "character_relationships.json"
    )

    open_windows: list[QWidget] = []

    pet_dir = settings.window.pet_dir or _kiosk_default_pet_dir(settings)
    speech_controller = _build_speech_controller(settings, pet_dir)
    voice_input = _build_voice_input(settings)
    voice_capture, stt_worker = voice_input if voice_input is not None else (None, None)

    settings_server = SettingsServer(settings_path, default_vault_paths(bootstrap))
    settings_server.start()

    def _on_quit() -> None:
        cancellation_gate.request_stop()
        chat_input.put(_SHUTDOWN)
        brain_thread.join(timeout=10.0)
        if brain_thread.is_alive():
            logger.warning("Brain 线程在 10 秒内未能正常退出，强制关闭应用")
        settings_server.stop()
        if speech_controller is not None:
            speech_controller.close()
        if voice_input is not None:
            _, worker = voice_input
            worker.stop()
            worker.wait(3000)
        vault.close()
        app.quit()

    def _queue_message(text: str) -> None:
        message_queue.push(text)

    def _open_settings() -> SettingsPanel:
        return _open_settings_panel(settings_path, open_windows, vault=vault)

    # ``window`` 在下面这些闭包创建时还未赋值，但都只在用户真正触发对应操作时才会被
    # 调用（必然晚于 `window = OverlayWindow(...)` 的赋值）——跟 main.py 里同样写法的
    # 依据完全一致：Python 闭包按名字在调用时从 `main()` 的作用域里取值。
    actions = PetActions(
        talk=chat_input.put,
        queue_message=_queue_message,
        open_settings=_open_settings,
        open_memory=lambda: _open_memory_panel(memory_system, open_windows),
        open_characters=lambda: _open_character_gallery(
            window,
            settings_path,
            open_windows,
            vault=vault,
            speech_controller=speech_controller,
            relationship_store=relationship_store,
        ),
        open_recollections=lambda: _open_recollection_gallery(memory_system, open_windows),
        toggle_proactive=lambda enabled: chat_input.put(ProactiveToggleRequest(enabled=enabled)),
        quit=_on_quit,
    )

    def _open_kiosk_settings() -> None:
        # 每次打开都重新从磁盘读一次最新配置，而不是复用外层 `pet_dir` 局部变量——
        # 用户可能已经通过上一次打开的面板切换过角色，`pet_dir` 那份闭包捕获的值
        # 早就过期了，`main.py::_open_character_gallery` 用的是同一个"现读现用"模式。
        current_settings = AppSettings.load(settings_path)
        current_pet_dir = current_settings.window.pet_dir or _kiosk_default_pet_dir(
            current_settings
        )
        panel = KioskSettingsPanel(_assets_pets_dir(), current_pet_dir, settings_path)
        panel.character_switched.connect(
            lambda new_pet_dir: _on_character_switched(
                new_pet_dir,
                window,
                settings_path,
                speech_controller=speech_controller,
                vault=vault,
                relationship_store=relationship_store,
            )
        )
        panel.clone_requested.connect(
            lambda: _open_kiosk_clone_dialog(
                window,
                settings_path,
                open_windows,
                vault=vault,
                speech_controller=speech_controller,
                relationship_store=relationship_store,
            )
        )
        panel.quit_requested.connect(_on_quit)
        open_windows.append(panel)
        panel.show()

    window = OverlayWindow(
        pet_dir,
        kiosk=True,
        scale=settings.kiosk.character_scale,
        event_bus=event_bus,
        confirmation_gate=confirm_gate,
        cancellation_gate=cancellation_gate,
        actions=actions,
        speech_controller=speech_controller,
        voice_capture=voice_capture,
        stt_worker=stt_worker,
        growth_store=growth_store,
        on_kiosk_settings_requested=_open_kiosk_settings,
    )
    if settings.kiosk.rotate_90_clockwise:
        # 嵌进 QGraphicsScene 前先把顶层窗口标志清掉——OverlayWindow 在 kiosk 模式下
        # 默认带着 Qt::Window（为了让自己单独 showFullScreen() 时真正生效，见
        # overlay_window.py 的说明），但这里它不再是被直接显示的顶层窗口，RotatedContainer
        # 才是；两者都带 Qt::Window 会让 Qt 把 window 也单独实体化成一个看得见的顶层窗口，
        # 实测在真机上会看到两个叠在一起的窗口。
        window.setWindowFlags(Qt.WindowType.Widget)
        rotated = RotatedContainer(window)
        rotated.showFullScreen()
    else:
        window.showFullScreen()

    health_warnings = _startup_health_warnings(settings, bootstrap)
    if health_warnings:
        health_box = MessageBox("启动检查", "\n\n".join(health_warnings), window)
        health_box.hideCancelButton()
        health_box.exec()

    app.exec()


def main() -> None:
    """Kiosk 入口。

    直接 framebuffer 模式下没有桌面可以承接异常。运行期崩溃交给 systemd 重启；初始化阶段
    的可恢复错误则保留在本进程的全屏页面，用户可在修复配置后触摸“重试”。
    """
    try:
        _run_kiosk()
    except Exception as exc:
        raise SystemExit(_show_startup_failure(exc)) from exc


if __name__ == "__main__":
    main()
