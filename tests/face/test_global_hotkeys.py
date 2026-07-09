"""``translate_qt_shortcut_to_pynput`` 的 token 翻译与 ``GlobalHotKeyManager`` 换绑回归测试。

用一个假 ``GlobalHotKeys`` 顶替 pynput 真实的原生监听线程（记录构造参数，
``start``/``stop``/``join`` 均为空实现），思路上跟 test_speech_controller.py 里
``_no_real_worker_threads`` 这个 autouse fixture 一致——只是这里替换的是外部库的类，
不是自己的 ``QThread``，避免测试真的起一个原生键盘监听线程。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import pytest
from PySide6.QtWidgets import QApplication

from miku_on_desk.face.ui import global_hotkeys
from miku_on_desk.face.ui.global_hotkeys import (
    GlobalHotKeyManager,
    translate_qt_shortcut_to_pynput,
)


class _FakeGlobalHotKeys:
    """顶替 ``pynput.keyboard.GlobalHotKeys``：记录构造参数与调用次数，不起真实线程。"""

    instances: ClassVar[list[_FakeGlobalHotKeys]] = []

    def __init__(self, hotkeys: dict[str, Callable[[], None]]) -> None:
        self.hotkeys = hotkeys
        self.started = 0
        self.stopped = 0
        self.joined = 0
        _FakeGlobalHotKeys.instances.append(self)

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def join(self, timeout: float | None = None) -> None:
        self.joined += 1


@pytest.fixture(autouse=True)
def _fake_global_hotkeys(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeGlobalHotKeys.instances = []
    monkeypatch.setattr(global_hotkeys, "GlobalHotKeys", _FakeGlobalHotKeys)


# ---- translate_qt_shortcut_to_pynput ----


def test_translate_plain_letter_lowercases_single_char_token() -> None:
    assert translate_qt_shortcut_to_pynput("Y") == "y"


def test_translate_modifier_combo() -> None:
    assert translate_qt_shortcut_to_pynput("Ctrl+Shift+Y") == "<ctrl>+<shift>+y"


def test_translate_meta_modifier_maps_to_cmd_token() -> None:
    assert translate_qt_shortcut_to_pynput("Ctrl+Meta+A") == "<ctrl>+<cmd>+a"


@pytest.mark.parametrize(
    ("qt_token", "pynput_token"),
    [
        ("PgUp", "<page_up>"),
        ("PgDown", "<page_down>"),
        ("Del", "<delete>"),
        ("Ins", "<insert>"),
        ("Esc", "<esc>"),
        ("F12", "<f12>"),
    ],
)
def test_translate_special_key_tokens(qt_token: str, pynput_token: str) -> None:
    assert translate_qt_shortcut_to_pynput(f"Ctrl+{qt_token}") == f"<ctrl>+{pynput_token}"


def test_translate_unrecognized_token_raises_value_error() -> None:
    with pytest.raises(ValueError, match="PageUp"):
        translate_qt_shortcut_to_pynput("Ctrl+PageUp")


def test_translate_empty_string_raises_value_error() -> None:
    with pytest.raises(ValueError):
        translate_qt_shortcut_to_pynput("")


# ---- GlobalHotKeyManager.rebind ----


def test_rebind_starts_listener_with_translated_bindings(qapp: QApplication) -> None:
    manager = GlobalHotKeyManager()

    manager.rebind({"open_chat": "Ctrl+Shift+M"})

    assert len(_FakeGlobalHotKeys.instances) == 1
    listener = _FakeGlobalHotKeys.instances[0]
    assert listener.started == 1
    assert list(listener.hotkeys.keys()) == ["<ctrl>+<shift>+m"]


def test_rebind_skips_invalid_binding_without_dropping_valid_ones(
    qapp: QApplication,
) -> None:
    manager = GlobalHotKeyManager()

    manager.rebind({"open_chat": "Ctrl+PageUp", "confirm_yes": "Ctrl+Shift+Y"})

    listener = _FakeGlobalHotKeys.instances[0]
    assert list(listener.hotkeys.keys()) == ["<ctrl>+<shift>+y"]


def test_rebind_with_only_invalid_bindings_does_not_start_listener(
    qapp: QApplication,
) -> None:
    manager = GlobalHotKeyManager()

    manager.rebind({"open_chat": "Ctrl+PageUp"})

    assert _FakeGlobalHotKeys.instances == []


def test_rebind_colliding_combos_keeps_one_action_without_crashing(
    qapp: QApplication,
) -> None:
    manager = GlobalHotKeyManager()

    manager.rebind({"open_chat": "Ctrl+Shift+M", "confirm_yes": "Ctrl+Shift+M"})

    listener = _FakeGlobalHotKeys.instances[0]
    assert len(listener.hotkeys) == 1


def test_rebind_twice_stops_and_joins_previous_listener(qapp: QApplication) -> None:
    manager = GlobalHotKeyManager()
    manager.rebind({"open_chat": "Ctrl+Shift+M"})
    first_listener = _FakeGlobalHotKeys.instances[0]

    manager.rebind({"confirm_yes": "Ctrl+Shift+Y"})

    assert first_listener.stopped == 1
    assert first_listener.joined == 1
    assert len(_FakeGlobalHotKeys.instances) == 2


def test_close_stops_and_joins_listener(qapp: QApplication) -> None:
    manager = GlobalHotKeyManager()
    manager.rebind({"open_chat": "Ctrl+Shift+M"})
    listener = _FakeGlobalHotKeys.instances[0]

    manager.close()

    assert listener.stopped == 1
    assert listener.joined == 1
