"""把待合成文本里"念出来是噪音"的字符清掉：emoji/图形符号 + Markdown 装饰符。

为什么需要：LLM 回复常夹带 emoji（😊👍）、颜文字符号、以及 ``**加粗**``/``# 标题``/
``` `代码` ``` 这类 Markdown 标记。直接丢给 TTS，轻则读出"星号星号"，重则引擎对满屏
符号发音怪异。这里在断句之后、送去合成之前把这些字符剔除，只保留真正要念的文字。

保守起见只删两类：Unicode 的"其它符号/修饰符号"（``So``/``Sk``，涵盖绝大多数 emoji、
箭头、™©® 等）与显式的 Markdown 装饰符；数学符号（``+ = < >`` 等）、货币、字母数字、
正常标点一律保留，避免把"3 > 2"这类有意义的内容也吞掉。链接 ``[文字](url)`` 不在此
处理范围内（会牵涉 URL 改写，另议）。

纯函数、无副作用、无 Qt/网络依赖，便于单测。
"""

from __future__ import annotations

import unicodedata

# Markdown 装饰符：念出来是噪音，删掉后内部文字保留（``**粗**`` → ``粗``）。
_MARKDOWN_MARKUP = frozenset("*_`#~|>")


def _is_emoji_or_symbol(char: str) -> bool:
    """判断字符是否为 emoji / 图形符号 / emoji 组合用的不可见连接符。"""
    code = ord(char)
    # 变体选择符（U+FE0F 等）、零宽连接符、肤色修饰、区域指示符（国旗）——
    # 这些本身分类各异（Mn/Cf/So），显式按码位范围兜住，避免漏删 emoji 组合件。
    if 0xFE00 <= code <= 0xFE0F:
        return True
    if code == 0x200D:
        return True
    if 0x1F3FB <= code <= 0x1F3FF:
        return True
    if 0x1F1E6 <= code <= 0x1F1FF:
        return True
    # So=其它符号（多数 emoji、箭头、™©®…），Sk=修饰符号（多为颜文字用符号）。
    return unicodedata.category(char) in ("So", "Sk")


def sanitize_for_speech(text: str) -> str:
    """剔除 emoji/图形符号与 Markdown 装饰符，合并多余空白后返回。

    若清理后只剩空白（如整句只有 ``😊👍``），返回空串——调用方据此跳过该句、不做合成。
    """
    kept = [
        char
        for char in text
        if char not in _MARKDOWN_MARKUP and not _is_emoji_or_symbol(char)
    ]
    # 删字符后可能留下连续空白（如"谢谢 😊 大家"→"谢谢  大家"）；split/join 顺带收尾空白。
    return " ".join("".join(kept).split())
