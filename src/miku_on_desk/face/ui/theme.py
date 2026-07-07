"""Miku 主题配色常量、间距/圆角设计令牌与 QFluentWidgets 全局主题装配。"""

from __future__ import annotations

from PySide6.QtGui import QColor

TEAL_MAIN = "#8fdac6"
TEAL_DARK = "#50b9ac"
PINK_ACCENT = "#fc7ea2"

# 间距令牌（4px 网格）
SPACING_XXS = 2
SPACING_XS = 4
SPACING_SM = 8
SPACING_MD = 12
SPACING_LG = 16
SPACING_XL = 24

# 圆角令牌
RADIUS_SM = 4
RADIUS_MD = 6
RADIUS_LG = 8
RADIUS_XL = 16  # 药丸形输入框专用（chat_popup.py）

# 语义色令牌
ERROR_COLOR = "#e05a5a"
WARNING_COLOR = "#e0a95a"
SUCCESS_COLOR = TEAL_DARK
HOVER_COLOR = TEAL_DARK
PRESSED_COLOR = TEAL_MAIN
PLACEHOLDER_BG = "#3a3a3a"


def qcolor(hex_value: str, alpha: int = 255) -> QColor:
    color = QColor(hex_value)
    color.setAlpha(alpha)
    return color


def border_qss(
    color: str = TEAL_DARK, *, width: int = 2, radius: int = RADIUS_LG, style: str = "solid"
) -> str:
    """统一散落在各处的 ``border: ...; border-radius: ...;`` 拼接写法。"""
    return f"border: {width}px {style} {color}; border-radius: {radius}px;"


def apply_fluent_theme() -> None:
    from qfluentwidgets import Theme, setTheme, setThemeColor

    setTheme(Theme.AUTO)
    setThemeColor(TEAL_MAIN)
