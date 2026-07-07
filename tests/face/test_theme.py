"""theme.py 里设计令牌（间距/圆角/语义色/border_qss helper）的回归测试，
确保 Phase 1 起各文件的字面量换令牌重构有一个可逐字节核对的契约。
"""

from __future__ import annotations

from miku_on_desk.face.ui.theme import (
    ERROR_COLOR,
    PLACEHOLDER_BG,
    RADIUS_LG,
    RADIUS_MD,
    RADIUS_SM,
    RADIUS_XL,
    SPACING_LG,
    SPACING_MD,
    SPACING_SM,
    SPACING_XL,
    SPACING_XS,
    SPACING_XXS,
    SUCCESS_COLOR,
    TEAL_DARK,
    TEAL_MAIN,
    WARNING_COLOR,
    border_qss,
    qcolor,
)


def test_spacing_tokens_are_monotonically_increasing() -> None:
    spacing = [SPACING_XXS, SPACING_XS, SPACING_SM, SPACING_MD, SPACING_LG, SPACING_XL]

    assert spacing == sorted(spacing)
    assert len(set(spacing)) == len(spacing)


def test_radius_tokens_match_existing_hardcoded_values() -> None:
    assert RADIUS_SM == 4
    assert RADIUS_MD == 6
    assert RADIUS_LG == 8
    assert RADIUS_XL == 16


def test_semantic_color_tokens_match_existing_hardcoded_values() -> None:
    assert ERROR_COLOR == "#e05a5a"
    assert WARNING_COLOR == "#e0a95a"
    assert SUCCESS_COLOR == TEAL_DARK
    assert PLACEHOLDER_BG == "#3a3a3a"


def test_border_qss_renders_default_solid_border() -> None:
    assert border_qss(TEAL_DARK) == "border: 2px solid #50b9ac; border-radius: 8px;"


def test_border_qss_supports_custom_width_radius_and_style() -> None:
    assert (
        border_qss(TEAL_MAIN, width=1, radius=RADIUS_MD, style="dashed")
        == "border: 1px dashed #8fdac6; border-radius: 6px;"
    )


def test_qcolor_still_applies_alpha() -> None:
    color = qcolor(TEAL_MAIN, alpha=128)

    assert color.name() == TEAL_MAIN
    assert color.alpha() == 128
