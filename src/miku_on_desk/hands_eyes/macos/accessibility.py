"""macOS accessibility tree 枚举：用 pyobjc 的 ApplicationServices 绑定 AXUIElement C API。

坑点：按钮等控件的可读标签几乎都挂在 kAXDescriptionAttribute 而非 kAXTitleAttribute
（实测系统计算器的数字/运算符按钮 AXTitle 全部是 None，标签在 AXDescription 里），
因此取标签要 description 优先、title 兜底。另外调用方所在进程必须已被系统「辅助功能」
权限授权（AXIsProcessTrusted），否则以下所有 AXUIElementCopyAttributeValue 调用都会
静默失败并返回空结果，而不是抛异常，容易被误判为「压根没有可交互元素」。
"""

from __future__ import annotations

from typing import Any

import AppKit
import ApplicationServices as AS
import Quartz

from miku_on_desk.hands_eyes.backend import ForegroundAppInfo, UIElement

_AX_LINK_ROLE = "AXLink"  # pyobjc 的 ApplicationServices 绑定未导出 kAXLinkRole 常量，只能用字面值
# 同样未导出的行/单元格角色常量：微信联系人列表这类自绘列表控件常用 AXRow/AXCell/AXOutlineRow
# 承载可点击的联系人行，只能用字面值兜底。
_AX_ROW_ROLE = "AXRow"
_AX_CELL_ROLE = "AXCell"
_AX_OUTLINE_ROW_ROLE = "AXOutlineRow"

_INTERACTIVE_ROLES = frozenset(
    {
        AS.kAXButtonRole,
        AS.kAXCheckBoxRole,
        AS.kAXRadioButtonRole,
        AS.kAXTextFieldRole,
        AS.kAXTextAreaRole,
        AS.kAXMenuItemRole,
        AS.kAXPopUpButtonRole,
        _AX_LINK_ROLE,
        AS.kAXSliderRole,
        _AX_ROW_ROLE,
        _AX_CELL_ROLE,
        _AX_OUTLINE_ROW_ROLE,
    }
)


def is_accessibility_trusted() -> bool:
    return bool(AS.AXIsProcessTrusted())


def _get_attr(element: Any, attribute: str) -> Any:
    error, value = AS.AXUIElementCopyAttributeValue(element, attribute, None)
    return None if error != 0 else value


def _unwrap_axvalue(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    value_type = AS.AXValueGetType(value)
    if value_type == AS.kAXValueTypeCGPoint:
        ok, point = AS.AXValueGetValue(value, value_type, None)
        return (point.x, point.y) if ok else None
    if value_type == AS.kAXValueTypeCGSize:
        ok, size = AS.AXValueGetValue(value, value_type, None)
        return (size.width, size.height) if ok else None
    return None


def _walk(element: Any, results: list[UIElement]) -> None:
    role = _get_attr(element, AS.kAXRoleAttribute)
    if role in _INTERACTIVE_ROLES:
        position = _unwrap_axvalue(_get_attr(element, AS.kAXPositionAttribute))
        size = _unwrap_axvalue(_get_attr(element, AS.kAXSizeAttribute))
        if position is not None and size is not None:
            label = (
                _get_attr(element, AS.kAXDescriptionAttribute)
                or _get_attr(element, AS.kAXTitleAttribute)
                or ""
            )
            results.append(
                UIElement(
                    role=str(role),
                    label=str(label),
                    center_x=int(position[0] + size[0] / 2),
                    center_y=int(position[1] + size[1] / 2),
                    width=int(size[0]),
                    height=int(size[1]),
                )
            )
    for child in _get_attr(element, AS.kAXChildrenAttribute) or []:
        _walk(child, results)


def list_elements(pid: int) -> list[UIElement]:
    """枚举指定进程所有窗口下的可交互元素，坐标已是全局屏幕像素坐标。"""
    app_ref = AS.AXUIElementCreateApplication(pid)
    windows = _get_attr(app_ref, AS.kAXWindowsAttribute) or []
    results: list[UIElement] = []
    for window in windows:
        _walk(window, results)
    return results



def get_window_bounds(pid: int) -> tuple[int, int, int, int] | None:
    """返回指定进程前台窗口的全局桌面坐标边界 (x, y, width, height)；取不到时返回 None。"""
    app_ref = AS.AXUIElementCreateApplication(pid)
    window = _get_attr(app_ref, AS.kAXMainWindowAttribute)
    if window is None:
        windows = _get_attr(app_ref, AS.kAXWindowsAttribute) or []
        window = windows[0] if windows else None
    if window is None:
        return None
    position = _unwrap_axvalue(_get_attr(window, AS.kAXPositionAttribute))
    size = _unwrap_axvalue(_get_attr(window, AS.kAXSizeAttribute))
    if position is None or size is None:
        return None
    return (round(position[0]), round(position[1]), round(size[0]), round(size[1]))


def get_idle_seconds() -> float:
    """距离上一次真实用户输入（键盘/鼠标）过去了多少秒。"""
    return float(
        Quartz.CGEventSourceSecondsSinceLastEventType(
            Quartz.kCGEventSourceStateHIDSystemState, Quartz.kCGAnyInputEventType
        )
    )


def get_foreground_app_info() -> ForegroundAppInfo | None:
    """当前前台应用名 + 聚焦窗口标题；应用名或窗口标题任一取不到都整体返回 None。"""
    app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None:
        return None
    app_name = app.localizedName()
    if not app_name:
        return None
    app_ref = AS.AXUIElementCreateApplication(app.processIdentifier())
    window = _get_attr(app_ref, AS.kAXFocusedWindowAttribute)
    title = _get_attr(window, AS.kAXTitleAttribute) if window is not None else None
    if not title:
        return None
    return ForegroundAppInfo(app_name=str(app_name), window_title=str(title))
