"""``sanitize_for_speech`` 过滤 emoji/符号/Markdown 装饰符的回归测试。"""

from __future__ import annotations

from miku_on_desk.brain.tts.text_sanitizer import sanitize_for_speech


def test_plain_text_is_unchanged() -> None:
    assert sanitize_for_speech("今天天气不错。") == "今天天气不错。"


def test_strips_emoji_and_collapses_leftover_space() -> None:
    assert sanitize_for_speech("谢谢 😊 大家") == "谢谢 大家"


def test_strips_emoji_between_cjk_without_gap() -> None:
    assert sanitize_for_speech("好耶🎉太棒了") == "好耶太棒了"


def test_strips_zwj_and_variation_selector_emoji_sequences() -> None:
    # 👨‍👩‍👧 是 ZWJ 组合，❤️ 带变体选择符——组合件都应被清掉。
    assert sanitize_for_speech("一家人👨‍👩‍👧和爱心❤️") == "一家人和爱心"


def test_strips_flag_regional_indicators() -> None:
    assert sanitize_for_speech("来自🇯🇵日本") == "来自日本"


def test_strips_markdown_decoration_keeps_inner_text() -> None:
    # 装饰符删掉、内部文字保留；``# `` 去掉井号后留下的单个空格无害，不强行合并。
    assert sanitize_for_speech("这是**加粗**和`代码`和# 标题") == "这是加粗和代码和 标题"


def test_keeps_meaningful_math_symbols() -> None:
    # 数学/比较符号（Sm）是有意义的内容，不能吞掉——否则"3 > 2"会被毁掉。
    assert sanitize_for_speech("3 + 2 = 5") == "3 + 2 = 5"


def test_all_emoji_sentence_becomes_empty() -> None:
    # 整句只有表情 → 空串，调用方据此跳过合成。
    assert sanitize_for_speech("😊👍🎉") == ""


def test_strips_trademark_and_pictographic_symbols() -> None:
    # ™©® 属 So（其它符号），应被清掉。
    assert sanitize_for_speech("产品™©®上市") == "产品上市"


def test_whitespace_only_input_returns_empty() -> None:
    assert sanitize_for_speech("   \t  ") == ""
