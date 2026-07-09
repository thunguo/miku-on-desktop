"""为「克隆」向导生成一段约 30 秒的朗读文本，直接复用现有 LLM 调用链路，不新开一套。"""

from __future__ import annotations

from miku_on_desk.brain.model_router import ModelRouter
from miku_on_desk.brain.providers.base import Message, Provider
from miku_on_desk.config.settings import ModelTier, ProviderName

_SYSTEM_PROMPT = (
    "你是一个为语音克隆准备朗读文本的助手。只输出朗读文本本身，不要有任何前后缀、"
    "解释或markdown标记。"
)

_USER_PROMPT_TEMPLATE = """\
请写一段大约 90-120 个字的中文朗读文本，供用户对着麦克风朗读约 30 秒用于声音克隆取样。

要求：
- 口语化、自然，混合陈述句、疑问句、感叹句，让朗读者的语气有起伏变化。
- 不要包含数字编号、引号、括号或任何标点之外的符号。
- 可以带一点这个角色的设定风味，但不要偏题太远：{description}
- 只输出朗读文本本身，不要加任何说明、标题或前后缀。
"""


class ReadingScriptError(Exception):
    """朗读文本生成失败（模型返回空内容等）。"""


async def generate_reading_script(
    *,
    description: str,
    router: ModelRouter,
    providers: dict[ProviderName, Provider],
    tier: ModelTier = ModelTier.MINI,
) -> str:
    """生成一段供朗读录音使用的文本；``NoModelAvailableError`` 原样透传，不吞掉。"""
    resolved = router.resolve(tier)
    provider = providers[resolved.provider]
    result = await provider.stream(
        model=resolved.model_id,
        system=_SYSTEM_PROMPT,
        messages=[
            Message(role="user", content=_USER_PROMPT_TEMPLATE.format(description=description))
        ],
        tools=[],
    )
    text = result.content.strip()
    if not text:
        raise ReadingScriptError(result.raw_error or result.error or "模型返回了空的朗读文本")
    return text
