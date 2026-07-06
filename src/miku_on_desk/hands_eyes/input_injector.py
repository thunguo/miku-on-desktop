"""跨平台鼠标/键盘输入注入：pynput 已经处理了 Windows/macOS 的差异，不需要分平台实现。

文本输入走剪贴板整段粘贴，而不是逐字符 ``keyboard.type()``：逐字符按键对中文/emoji 等多
字节字符容易丢字符或触发某些应用的输入速率限制，剪贴板+粘贴键both更快也更可靠。粘贴后把
剪贴板还原成之前的内容，避免污染用户自己的剪贴板历史。

按键组合的 mac Control→Command 重映射：只在组合里出现 ctrl 且没有 cmd、组合长度 ≥2、且
至少有一个非修饰键不在 ``_MAC_CTRL_KEEP``（mac 原生 Control 已有含义的键，如方向键/Tab/Esc）
时才重映射——这样能保留 mac 原生 Ctrl+Tab 之类的快捷键，同时把 Windows/Linux 风格的 Ctrl+C
正确翻译成 Cmd+C。
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence

import pyperclip
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

_mouse = MouseController()
_keyboard = KeyboardController()

_KEY_MAP: dict[str, Key] = {
    "alt": Key.alt,
    "alt_r": Key.alt_r,
    "backspace": Key.backspace,
    "caps_lock": Key.caps_lock,
    "cmd": Key.cmd,
    "command": Key.cmd,
    "win": Key.cmd,
    "super": Key.cmd,
    "cmd_r": Key.cmd_r,
    "ctrl": Key.ctrl,
    "control": Key.ctrl,
    "ctrl_r": Key.ctrl_r,
    "delete": Key.delete,
    "down": Key.down,
    "end": Key.end,
    "enter": Key.enter,
    "return": Key.enter,
    "esc": Key.esc,
    "escape": Key.esc,
    "home": Key.home,
    "left": Key.left,
    "page_down": Key.page_down,
    "page_up": Key.page_up,
    "right": Key.right,
    "shift": Key.shift,
    "shift_r": Key.shift_r,
    "space": Key.space,
    "tab": Key.tab,
    "up": Key.up,
    **{f"f{i}": getattr(Key, f"f{i}") for i in range(1, 21)},
}

_MODIFIER_NAMES = {"ctrl", "ctrl_r", "cmd", "cmd_r", "alt", "alt_r", "shift", "shift_r"}

# 粘贴键投递后到恢复剪贴板前的等待：给目标应用留时间真正读取剪贴板，避免应用还没消费这次
# 粘贴、剪贴板就已经被恢复成用户的旧内容（目标应用刚被 open_app 拉起、窗口还没就绪时最容易
# 触发这个竞态）。
_PASTE_SETTLE_DELAY_S = 0.3

_MAC_CTRL_KEEP = {
    "up",
    "down",
    "left",
    "right",
    "tab",
    "space",
    "page_up",
    "page_down",
    "home",
    "end",
    "esc",
    "escape",
    "enter",
    "return",
    "delete",
    "backspace",
} | {f"f{i}" for i in range(1, 21)}


def _remap_ctrl_to_cmd_for_mac(keys: Sequence[str]) -> list[str]:
    lowered = [k.lower() for k in keys]
    if sys.platform != "darwin" or "ctrl" not in lowered or "cmd" in lowered or len(lowered) < 2:
        return lowered
    has_real_key = any(k not in _MODIFIER_NAMES and k not in _MAC_CTRL_KEEP for k in lowered)
    if not has_real_key:
        return lowered
    return ["cmd" if k == "ctrl" else k for k in lowered]


def _resolve_key(name: str) -> Key | str:
    mapped = _KEY_MAP.get(name.lower())
    if mapped is not None:
        return mapped
    if len(name) == 1:
        return name
    raise ValueError(f"未知的按键名称：{name}")


def click(x: int, y: int) -> None:
    _mouse.position = (x, y)
    _mouse.click(Button.left, 1)


def press_keys(keys: Sequence[str]) -> None:
    if not keys:
        raise ValueError("按键组合不能为空")
    resolved = [_resolve_key(k) for k in _remap_ctrl_to_cmd_for_mac(keys)]
    modifiers, main_key = resolved[:-1], resolved[-1]
    if modifiers:
        with _keyboard.pressed(*modifiers):
            _keyboard.tap(main_key)
    else:
        _keyboard.tap(main_key)


def type_text(text: str) -> None:
    previous = pyperclip.paste()
    try:
        pyperclip.copy(text)
        time.sleep(0.05)
        press_keys(["cmd" if sys.platform == "darwin" else "ctrl", "v"])
        time.sleep(_PASTE_SETTLE_DELAY_S)
    finally:
        pyperclip.copy(previous)
