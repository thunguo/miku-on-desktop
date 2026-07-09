"""系统级全局热键：跨应用、无需 Miku 窗口获得焦点即可触发。

pynput 的 HotKey DSL（``"<ctrl>+<shift>+y"``）跟 Qt 的 QKeySequence PortableText
（``"Ctrl+Shift+Y"``）语法不同，``translate_qt_shortcut_to_pynput`` 做一次性 token 翻译。

热键回调发生在 pynput 的监听线程（不是 Qt 主线程）；``hotkey_triggered`` 信号在那个线程里
emit，Qt 的 AutoConnection 会自动把连接的槽调度到接收者所在线程（构造
``GlobalHotKeyManager`` 时所在的 UI 主线程）——跟 speech_controller.py 里 ``_SynthWorker``
向 UI 线程回传音频 chunk 是同一个模式，不需要手写锁/队列。
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable

from pynput.keyboard import GlobalHotKeys, HotKey
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

_QT_MODIFIER_TOKENS: dict[str, str] = {
    "Ctrl": "<ctrl>",
    "Alt": "<alt>",
    "Shift": "<shift>",
    "Meta": "<cmd>",
}

# 下列拼写均用真实 QKeySequence（由 Qt.Key 常量构造，即 QKeySequenceEdit 捕获真实按键时
# 产出的同一路径）直接验证过，不是猜测；PageUp/PageDown 尤其容易搞错——
# QKeySequence("Ctrl+PageUp") 解析失败返回空串，但真实按键产出的是 "Ctrl+PgUp"。
_QT_SPECIAL_KEY_TOKENS: dict[str, str] = {
    "Esc": "<esc>",
    "Tab": "<tab>",
    "Backspace": "<backspace>",
    "Return": "<enter>",
    "Enter": "<enter>",
    "Space": "<space>",
    "Del": "<delete>",
    "Ins": "<insert>",
    "Home": "<home>",
    "End": "<end>",
    "PgUp": "<page_up>",
    "PgDown": "<page_down>",
    "Up": "<up>",
    "Down": "<down>",
    "Left": "<left>",
    "Right": "<right>",
    "CapsLock": "<caps_lock>",
    "Menu": "<menu>",
    "Pause": "<pause>",
    "Print": "<print_screen>",
    "ScrollLock": "<scroll_lock>",
    "NumLock": "<num_lock>",
    **{f"F{i}": f"<f{i}>" for i in range(1, 21)},  # pynput 的 Key 枚举只到 f20
}


def translate_qt_shortcut_to_pynput(qt_sequence: str) -> str:
    """把 Qt PortableText（如 ``"Ctrl+Shift+Y"``）翻译成 pynput 的 HotKey DSL
    （如 ``"<ctrl>+<shift>+y"``）。无法识别的 token 会 raise ``ValueError``，调用方
    负责捕获并跳过这一条绑定。
    """
    if not qt_sequence:
        raise ValueError("空快捷键")
    pynput_tokens: list[str] = []
    for token in qt_sequence.split("+"):
        if token in _QT_MODIFIER_TOKENS:
            pynput_tokens.append(_QT_MODIFIER_TOKENS[token])
        elif token in _QT_SPECIAL_KEY_TOKENS:
            pynput_tokens.append(_QT_SPECIAL_KEY_TOKENS[token])
        elif len(token) == 1:
            pynput_tokens.append(token.lower())
        else:
            raise ValueError(f"无法识别的按键 token: {token!r}（来自 {qt_sequence!r}）")
    return "+".join(pynput_tokens)


class GlobalHotKeyManager(QObject):
    """管理一组"动作名 -> QKeySequence 字符串"的系统级全局热键绑定。"""

    hotkey_triggered = Signal(str)  # 动作名，如 "open_chat"/"confirm_yes"/"confirm_no"

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._listener: GlobalHotKeys | None = None

    def rebind(self, bindings: dict[str, str]) -> None:
        """停掉旧监听线程（若有），用新绑定表建一个新的 GlobalHotKeys 并启动。"""
        self._stop_listener()

        pynput_hotkeys: dict[str, Callable[[], None]] = {}
        action_by_combo: dict[str, str] = {}
        for action, qt_sequence in bindings.items():
            try:
                pynput_sequence = translate_qt_shortcut_to_pynput(qt_sequence)
                HotKey.parse(pynput_sequence)
            except ValueError:
                logger.warning("跳过无法解析的快捷键绑定：%s=%r", action, qt_sequence)
                continue
            if pynput_sequence in action_by_combo:
                logger.warning(
                    "快捷键 %r 同时绑定给 %s 和 %s，后者生效",
                    qt_sequence,
                    action_by_combo[pynput_sequence],
                    action,
                )
            action_by_combo[pynput_sequence] = action
            pynput_hotkeys[pynput_sequence] = functools.partial(self._emit, action)

        if not pynput_hotkeys:
            return
        self._listener = GlobalHotKeys(pynput_hotkeys)
        self._listener.start()

    def _emit(self, action: str) -> None:
        self.hotkey_triggered.emit(action)

    def _stop_listener(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener.join(timeout=1.0)
            self._listener = None

    def close(self) -> None:
        """应用退出时调用，对称于 speech_controller.close() 等既有约定。"""
        self._stop_listener()
