"""Windows accessibility tree 枚举：基于 uiautomation（UIA 的 Python 封装）。

本模块依据 uiautomation 库的公开 API 编写，无法在当前 macOS 开发机上实际运行验证，
交付前必须由用户在真实 Windows 机器上跑通验证。

坑点：uiautomation 的 ``Control.Name`` 已经是 UIA 计算好的可读名称，不像 macOS 那样
需要在 title/description 之间猜——不必额外兜底。顶层窗口枚举必须从桌面根控件
（``GetRootControl``）按 ``ProcessId`` 过滤子窗口开始；直接对某个窗口控件递归
``GetChildren()`` 遍历隐藏/离屏元素在 UIA 里代价很高，用 ``IsOffscreen`` 提前剪掉。
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

import psutil
import uiautomation as auto

from miku_on_desk.hands_eyes.backend import ForegroundAppInfo, UIElement

_INTERACTIVE_CONTROL_TYPES = frozenset(
    {
        auto.ControlType.ButtonControl,
        auto.ControlType.CheckBoxControl,
        auto.ControlType.RadioButtonControl,
        auto.ControlType.EditControl,
        auto.ControlType.ComboBoxControl,
        auto.ControlType.MenuItemControl,
        auto.ControlType.HyperlinkControl,
        auto.ControlType.SliderControl,
        auto.ControlType.TabItemControl,
        auto.ControlType.ListItemControl,
        auto.ControlType.DataItemControl,
    }
)


def _walk(control: auto.Control, results: list[UIElement]) -> None:
    if control.IsOffscreen:
        return
    if control.ControlType in _INTERACTIVE_CONTROL_TYPES:
        rect = control.BoundingRectangle
        if rect.width() > 0 and rect.height() > 0:
            results.append(
                UIElement(
                    role=str(control.ControlTypeName),
                    label=str(control.Name or ""),
                    center_x=rect.xcenter(),
                    center_y=rect.ycenter(),
                    width=rect.width(),
                    height=rect.height(),
                )
            )
    for child in control.GetChildren():
        _walk(child, results)


def list_elements(pid: int) -> list[UIElement]:
    """枚举指定进程所有顶层窗口下的可交互元素，坐标已是全局屏幕像素坐标。"""
    results: list[UIElement] = []
    root = auto.GetRootControl()
    for window in root.GetChildren():
        if window.ProcessId != pid:
            continue
        _walk(window, results)
    return results



def get_window_bounds(pid: int) -> tuple[int, int, int, int] | None:
    """返回指定进程前台窗口的全局桌面坐标边界 (x, y, width, height)；取不到时返回 None。"""
    root = auto.GetRootControl()
    for window in root.GetChildren():
        if window.ProcessId != pid:
            continue
        rect = window.BoundingRectangle
        return (rect.left, rect.top, rect.width(), rect.height())
    return None


class _LastInputInfo(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


def get_idle_seconds() -> float:
    """距离上一次真实用户输入（键盘/鼠标）过去了多少秒。"""
    info = _LastInputInfo()
    info.cbSize = ctypes.sizeof(_LastInputInfo)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):  # type: ignore[attr-defined]
        return 0.0
    millis_idle = ctypes.windll.kernel32.GetTickCount() - info.dwTime  # type: ignore[attr-defined]
    return float(millis_idle) / 1000.0


def get_foreground_app_info() -> ForegroundAppInfo | None:
    """当前前台应用名 + 窗口标题；应用名或窗口标题任一取不到都整体返回 None。"""
    hwnd = ctypes.windll.user32.GetForegroundWindow()  # type: ignore[attr-defined]
    if not hwnd:
        return None
    pid = wintypes.DWORD()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))  # type: ignore[attr-defined]
    try:
        app_name = psutil.Process(pid.value).name()
    except psutil.Error:
        return None
    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)  # type: ignore[attr-defined]
    if length == 0:
        return None
    buffer = ctypes.create_unicode_buffer(length + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buffer, length + 1)  # type: ignore[attr-defined]
    title = buffer.value
    if not title:
        return None
    return ForegroundAppInfo(app_name=app_name, window_title=title)
