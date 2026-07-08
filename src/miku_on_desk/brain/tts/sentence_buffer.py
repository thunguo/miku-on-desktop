"""流式文本的断句缓冲：把逐字到达的 ``ContentDelta`` 攒成整句再送去合成。

为什么需要它：TTS 按"整句"合成才自然，逐个 delta（可能只有一两个字）分别合成会产生
破碎、卡顿、语调断裂的语音。这里把到达的增量累积在缓冲里，只在遇到句末标点/换行时切出
完整句子返回，剩余不完整的部分继续留在缓冲里等后续增量。

纯逻辑、无副作用、无 Qt/网络依赖，便于单测覆盖各种断句边界。
"""

from __future__ import annotations

# 句末边界：中英文句末标点 + 分号 + 换行。逗号/顿号不算——按逗号切会产生大量过短片段，
# 反而让语音更碎。换行算边界，保证 Miku 分行输出（列表/多段）时能及时开口。
_SENTENCE_ENDINGS = frozenset("。！？!?…；;\n")


class SentenceBuffer:
    """累积流式文本，按句末边界切出完整句子。"""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """追加一段增量文本，返回本次因此凑成的完整句子（去空白后为空的片段被跳过）。

        可能返回 0 句（还没遇到边界）、1 句或多句（一次 delta 里含多个句末标点）。
        """
        self._buffer += text
        sentences: list[str] = []
        start = 0
        for index, char in enumerate(self._buffer):
            if char in _SENTENCE_ENDINGS:
                sentence = self._buffer[start : index + 1].strip()
                if sentence:
                    sentences.append(sentence)
                start = index + 1
        self._buffer = self._buffer[start:]
        return sentences

    def flush(self) -> str | None:
        """一轮回复结束时调用：返回缓冲里剩余的、没有句末标点收尾的最后一句（若非空白），
        并清空缓冲。返回 None 表示没有可读内容。
        """
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder or None
