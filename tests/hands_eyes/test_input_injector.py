"""input_injector 的回归测试：mac Ctrl→Cmd 重映射是纯逻辑，直接测试内部函数；
press_keys/type_text/click 这几个会产生真实鼠标键盘副作用的函数，把模块级 `_mouse`/
`_keyboard` 换成假对象，绝不在测试机上真的移动鼠标或模拟按键。
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from typing import Any

import pytest
from pynput.keyboard import Key

from miku_on_desk.hands_eyes import input_injector
from miku_on_desk.hands_eyes.input_injector import (
    _remap_ctrl_to_cmd_for_mac,
    _resolve_key,
    click,
    press_keys,
    type_text,
)


class _FakeKeyboard:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    @contextlib.contextmanager
    def pressed(self, *keys: Any) -> Iterator[None]:
        self.calls.append(("pressed", keys))
        yield

    def tap(self, key: Any) -> None:
        self.calls.append(("tap", (key,)))


class _FakeMouse:
    def __init__(self) -> None:
        self.position: tuple[int, int] | None = None
        self.clicks: list[tuple[Any, int]] = []

    def click(self, button: Any, count: int) -> None:
        self.clicks.append((button, count))


def test_remap_ctrl_to_cmd_translates_windows_style_shortcut_on_mac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    assert _remap_ctrl_to_cmd_for_mac(["ctrl", "c"]) == ["cmd", "c"]


def test_remap_ctrl_to_cmd_keeps_native_mac_control_combo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    assert _remap_ctrl_to_cmd_for_mac(["ctrl", "tab"]) == ["ctrl", "tab"]


def test_remap_ctrl_to_cmd_translates_when_real_key_mixed_with_keep_list_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    assert _remap_ctrl_to_cmd_for_mac(["ctrl", "shift", "c"]) == ["cmd", "shift", "c"]


def test_remap_ctrl_to_cmd_noop_when_cmd_already_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    assert _remap_ctrl_to_cmd_for_mac(["ctrl", "cmd", "c"]) == ["ctrl", "cmd", "c"]


def test_remap_ctrl_to_cmd_noop_for_single_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    assert _remap_ctrl_to_cmd_for_mac(["ctrl"]) == ["ctrl"]


def test_remap_ctrl_to_cmd_noop_on_non_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert _remap_ctrl_to_cmd_for_mac(["ctrl", "c"]) == ["ctrl", "c"]


def test_resolve_key_maps_known_name() -> None:
    assert _resolve_key("ctrl") is Key.ctrl


def test_resolve_key_passes_through_single_character() -> None:
    assert _resolve_key("a") == "a"


def test_resolve_key_rejects_unknown_multi_character_name() -> None:
    with pytest.raises(ValueError, match="未知的按键名称"):
        _resolve_key("not-a-real-key")


def test_press_keys_taps_main_key_under_modifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeKeyboard()
    monkeypatch.setattr(input_injector, "_keyboard", fake)
    monkeypatch.setattr(sys, "platform", "linux")

    press_keys(["ctrl", "c"])

    assert fake.calls == [("pressed", (Key.ctrl,)), ("tap", ("c",))]


def test_press_keys_single_key_taps_without_context_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeKeyboard()
    monkeypatch.setattr(input_injector, "_keyboard", fake)

    press_keys(["enter"])

    assert fake.calls == [("tap", (Key.enter,))]


def test_press_keys_rejects_empty_combo() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        press_keys([])


def test_click_moves_mouse_and_clicks(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeMouse()
    monkeypatch.setattr(input_injector, "_mouse", fake)

    click(12, 34)

    assert fake.position == (12, 34)
    assert len(fake.clicks) == 1


def test_type_text_copies_pastes_and_restores_clipboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_keyboard = _FakeKeyboard()
    monkeypatch.setattr(input_injector, "_keyboard", fake_keyboard)
    monkeypatch.setattr(sys, "platform", "linux")

    clipboard = {"value": "先前的剪贴板内容"}
    monkeypatch.setattr(input_injector.pyperclip, "paste", lambda: clipboard["value"])

    def _fake_copy(text: str) -> None:
        clipboard["value"] = text

    monkeypatch.setattr(input_injector.pyperclip, "copy", _fake_copy)

    type_text("你好世界")

    assert fake_keyboard.calls == [("pressed", (Key.ctrl,)), ("tap", ("v",))]
    assert clipboard["value"] == "先前的剪贴板内容"


def test_type_text_settles_after_paste_before_restoring_clipboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    monkeypatch.setattr(sys, "platform", "linux")

    clipboard = {"value": "先前的剪贴板内容"}
    monkeypatch.setattr(input_injector.pyperclip, "paste", lambda: clipboard["value"])

    def _fake_copy(text: str) -> None:
        order.append(f"copy:{text}")
        clipboard["value"] = text

    monkeypatch.setattr(input_injector.pyperclip, "copy", _fake_copy)

    def _fake_press_keys(keys: list[str]) -> None:
        order.append("press_keys")

    monkeypatch.setattr(input_injector, "press_keys", _fake_press_keys)

    def _fake_sleep(seconds: float) -> None:
        order.append(f"sleep:{seconds}")

    monkeypatch.setattr(input_injector.time, "sleep", _fake_sleep)

    type_text("你好世界")

    assert order == [
        "copy:你好世界",
        "sleep:0.05",
        "press_keys",
        f"sleep:{input_injector._PASTE_SETTLE_DELAY_S}",
        "copy:先前的剪贴板内容",
    ]


def test_type_text_restores_clipboard_even_if_paste_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    clipboard = {"value": "先前的剪贴板内容"}
    monkeypatch.setattr(input_injector.pyperclip, "paste", lambda: clipboard["value"])
    monkeypatch.setattr(input_injector.pyperclip, "copy", lambda text: clipboard.update(value=text))

    def _boom(_keys: list[str]) -> None:
        raise RuntimeError("模拟粘贴失败")

    monkeypatch.setattr(input_injector, "press_keys", _boom)

    with pytest.raises(RuntimeError, match="模拟粘贴失败"):
        type_text("你好")

    assert clipboard["value"] == "先前的剪贴板内容"
