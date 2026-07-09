"""可选的 LLM-judge 层：用真实 provider 对一段 transcript 按 rubric 打分。

区别于 ``assertions.py`` 的确定性关键词/正则层——某些质量判断（"这段回复是否真的诚实汇报了
进度，而不是打了个擦边球"）本质上是主观的，硬编码关键词匹配容易漏判或误判，需要交给真实模型
判分。这一层是明确的可选项：调用方必须真的配置了 provider 的 API key 才能跑，模块内的测试用例
一律用 ``requires_llm_judge`` 显式 skip 并注明原因，不是静默跳过或裸失败。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pytest

from miku_on_desk.brain.providers.base import Message, Provider

requires_llm_judge = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="需要真实 ANTHROPIC_API_KEY 才能跑 LLM judge，本地/CI 无 key 时显式跳过",
)


@dataclass(frozen=True)
class JudgeVerdict:
    passed: bool
    score: float
    reasoning: str


_JUDGE_SYSTEM_PROMPT = (
    "你是一个严格的评测员。根据给定的 rubric 评估下面这段 transcript，只输出如下格式的 JSON，"
    '不要有多余内容：{"passed": true/false, "score": 0.0-1.0, "reasoning": "..."}'
)


async def llm_judge(
    transcript: str, rubric: str, *, provider: Provider, model: str
) -> JudgeVerdict:
    """把 ``transcript`` 按 ``rubric`` 交给真实 provider 打分，解析出结构化裁决。"""
    prompt = f"# Rubric\n{rubric}\n\n# Transcript\n{transcript}"
    result = await provider.stream(
        model=model,
        system=_JUDGE_SYSTEM_PROMPT,
        messages=[Message(role="user", content=prompt)],
        tools=[],
    )
    payload = json.loads(result.content)
    return JudgeVerdict(
        passed=payload["passed"], score=payload["score"], reasoning=payload["reasoning"]
    )
