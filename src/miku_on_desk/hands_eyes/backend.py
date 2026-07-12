"""本机 OS 集成层的统一抽象接口。

accessibility 元素枚举因 macOS（AXUIElement）与 Windows（UIAutomation）原生 API
完全不同，必须分平台实现；点击/按键/输入注入用 pynput，在两个平台上是同一套代码，不需要
重复实现，统一放在 input_injector.py 里由两个具体 backend 共用。

``find_pid_by_name`` 用 psutil（跨平台）按进程名找 pid，是 accessibility 元素枚举
（按 pid 枚举窗口）与"打开应用"这两个操作之间的桥：先 open_app 唤起应用，再用进程名反查
pid 喂给 list_elements，调用方不需要自己知道任何平台特定的进程查找方式。
"""

from __future__ import annotations

import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import psutil

from miku_on_desk.brain.tools.registry import ToolExecutionError
from miku_on_desk.hands_eyes import input_injector


@dataclass(frozen=True)
class UIElement:
    """一个可交互的 accessibility 元素，坐标已是全局屏幕像素坐标，可直接喂给 click()。"""

    role: str
    label: str
    center_x: int
    center_y: int
    width: int
    height: int


def element_to_dict(element: UIElement) -> dict[str, Any]:
    return {
        "role": element.role,
        "text": element.label,
        "center_x": element.center_x,
        "center_y": element.center_y,
        "width": element.width,
        "height": element.height,
        "source": "accessibility",
    }


@dataclass(frozen=True)
class ForegroundAppInfo:
    """当前前台应用的名称与窗口标题，供主动交互调度器判断用户正在做什么。"""

    app_name: str
    window_title: str


class PlatformBackend(ABC):
    """一份具体实现对应一个操作系统，上层只通过这几个方法与本机 OS 交互。"""

    @abstractmethod
    def list_elements(self, pid: int) -> list[UIElement]:
        """枚举指定进程的 accessibility tree，返回可交互元素及其屏幕坐标。"""

    @abstractmethod
    def get_window_bounds(self, pid: int) -> tuple[int, int, int, int] | None:
        """返回指定进程前台窗口的全局桌面坐标边界 (x, y, width, height)；取不到时返回 None。"""

    @abstractmethod
    def open_app(self, name: str) -> None:
        """启动/唤起一个本机应用程序（如系统计算器、浏览器）。"""

    @abstractmethod
    def get_idle_seconds(self) -> float:
        """距离上一次真实用户输入（键盘/鼠标）过去了多少秒。"""

    @abstractmethod
    def get_foreground_app_info(self) -> ForegroundAppInfo | None:
        """当前前台应用名 + 窗口标题；取不到时返回 None。"""

    def click(self, x: int, y: int) -> None:
        input_injector.click(x, y)

    def type_text(self, text: str) -> None:
        input_injector.type_text(text)

    def press_keys(self, keys: Sequence[str]) -> None:
        input_injector.press_keys(keys)

    def find_pid_by_name(self, name: str) -> int | None:
        target = name.lower().removesuffix(".exe")
        for proc in psutil.process_iter(["pid", "name"]):
            proc_name = (proc.info.get("name") or "").lower().removesuffix(".exe")
            if proc_name == target:
                pid = proc.info.get("pid")
                if isinstance(pid, int):
                    return pid
        return None


class MacOSBackend(PlatformBackend):
    def list_elements(self, pid: int) -> list[UIElement]:
        from miku_on_desk.hands_eyes.macos import accessibility

        return accessibility.list_elements(pid)


    def get_window_bounds(self, pid: int) -> tuple[int, int, int, int] | None:
        from miku_on_desk.hands_eyes.macos import accessibility

        return accessibility.get_window_bounds(pid)

    def open_app(self, name: str) -> None:
        subprocess.run(["open", "-a", name], check=True)

    def get_idle_seconds(self) -> float:
        from miku_on_desk.hands_eyes.macos import accessibility

        return accessibility.get_idle_seconds()

    def get_foreground_app_info(self) -> ForegroundAppInfo | None:
        from miku_on_desk.hands_eyes.macos import accessibility

        return accessibility.get_foreground_app_info()


class WindowsBackend(PlatformBackend):
    def list_elements(self, pid: int) -> list[UIElement]:
        from miku_on_desk.hands_eyes.windows import accessibility

        return accessibility.list_elements(pid)


    def get_window_bounds(self, pid: int) -> tuple[int, int, int, int] | None:
        from miku_on_desk.hands_eyes.windows import accessibility

        return accessibility.get_window_bounds(pid)

    def open_app(self, name: str) -> None:
        """无法在当前 macOS 开发机上实际运行验证，交付前必须由用户在真实 Windows 机器上跑通。"""
        subprocess.run(["cmd", "/c", "start", "", name], check=True)

    def get_idle_seconds(self) -> float:
        from miku_on_desk.hands_eyes.windows import accessibility

        return accessibility.get_idle_seconds()

    def get_foreground_app_info(self) -> ForegroundAppInfo | None:
        from miku_on_desk.hands_eyes.windows import accessibility

        return accessibility.get_foreground_app_info()


class NullBackend(PlatformBackend):
    """Linux（树莓派 kiosk）上的占位实现：没有 accessibility API 可用，所有查询类方法
    返回"什么都没找到"而不是抛异常，让 ``computer_input``/``screen_analyze`` 工具能正常
    注册、被调用后安全地返回空结果，不拖垮整个 Brain 线程。``open_app`` 是唯一做不到
    "安全空实现"的操作——没有本机窗口系统可以唤起应用，直接抛 ``ToolExecutionError``
    让调用方知道这个操作在当前平台不支持，而不是伪装成功。"""

    def list_elements(self, pid: int) -> list[UIElement]:
        return []

    def get_window_bounds(self, pid: int) -> tuple[int, int, int, int] | None:
        return None

    def open_app(self, name: str) -> None:
        raise ToolExecutionError(f"当前平台不支持打开应用：{name}")

    def get_idle_seconds(self) -> float:
        return float("inf")

    def get_foreground_app_info(self) -> ForegroundAppInfo | None:
        return None


def create_platform_backend() -> PlatformBackend:
    """按当前操作系统选择具体实现。

    平台专属依赖（pyobjc/uiautomation）只在对应分支内导入，避免在错误的操作系统上
    因缺少依赖而导入失败——两者都是可选依赖，只在匹配的 sys_platform 下才会被安装。
    其余平台（Linux，即树莓派 kiosk）先用 ``NullBackend`` 占位，等 Phase 4 的
    ``CaptureCardBackend``（HDMI 采集卡 + USB HID 注入）实现后再按配置切换。
    """
    if sys.platform == "darwin":
        return MacOSBackend()
    if sys.platform == "win32":
        return WindowsBackend()
    return NullBackend()
