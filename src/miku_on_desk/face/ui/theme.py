"""Miku 主题配色常量与 QFluentWidgets 全局主题装配。"""

from __future__ import annotations

from PySide6.QtGui import QColor

TEAL_MAIN = "#8fdac6"
TEAL_DARK = "#50b9ac"
PINK_ACCENT = "#fc7ea2"


def qcolor(hex_value: str, alpha: int = 255) -> QColor:
    color = QColor(hex_value)
    color.setAlpha(alpha)
    return color


def apply_fluent_theme() -> None:
    from qfluentwidgets import Theme, setTheme, setThemeColor

    setTheme(Theme.AUTO)
    setThemeColor(TEAL_MAIN)
