"""SentenceBuffer 的断句边界回归测试：跨 delta 累积、多标点切分、残句 flush。"""

from __future__ import annotations

from miku_on_desk.brain.tts.sentence_buffer import SentenceBuffer


def test_feed_holds_incomplete_sentence_until_boundary_arrives() -> None:
    buffer = SentenceBuffer()

    assert buffer.feed("你好") == []
    assert buffer.feed("世界") == []
    assert buffer.feed("！") == ["你好世界！"]


def test_feed_splits_multiple_sentences_in_one_delta() -> None:
    buffer = SentenceBuffer()

    assert buffer.feed("第一句。第二句？第三") == ["第一句。", "第二句？"]
    assert buffer.flush() == "第三"


def test_feed_treats_newline_as_boundary() -> None:
    buffer = SentenceBuffer()

    assert buffer.feed("列表项一\n列表项二\n") == ["列表项一", "列表项二"]


def test_feed_skips_whitespace_only_fragments() -> None:
    buffer = SentenceBuffer()

    # 相邻边界之间只有空白（连续换行）时，中间的空片段应被丢弃而非产出空串句子。
    assert buffer.feed("好的。\n\n继续。") == ["好的。", "继续。"]


def test_flush_returns_remainder_and_clears_buffer() -> None:
    buffer = SentenceBuffer()

    buffer.feed("尾巴没有标点")
    assert buffer.flush() == "尾巴没有标点"
    assert buffer.flush() is None


def test_flush_returns_none_when_buffer_only_whitespace() -> None:
    buffer = SentenceBuffer()

    buffer.feed("   \n  ")
    # 换行是边界、会被 feed 消费掉，剩下的纯空白不构成可读句子。
    assert buffer.flush() is None
