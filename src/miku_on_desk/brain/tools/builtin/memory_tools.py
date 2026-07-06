"""``remember``/``recall`` 工具：让 LLM 在对话过程中主动记住/查询关于主人的长期信息。

与后台异步提取流水线（`brain/memory/extraction.py`）互补而非重复：提取流水线是"每隔几轮
对话或触发压缩时无条件跑一次提炼"，这里是"模型自己判断当下值得记的时候主动调用"——两者都
通过 `MemorySystem` 写入同一套语义事实存储，`extracted_by` 字段区分来源
（`tool:remember` vs. `llm:fast`），供 `face/ui/memory_panel.py` 区分"AI 主动记住"与
"提取流水线推断"的条目。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from miku_on_desk.brain.memory.system import MemorySystem
from miku_on_desk.brain.providers.base import ToolDefinition
from miku_on_desk.brain.tools.registry import (
    ToolExecutionError,
    ToolHandler,
    ToolRegistration,
    ToolRegistry,
)


class RememberInput(BaseModel):
    key: str
    value: str


class RecallInput(BaseModel):
    query: str


def _make_remember_handler(system: MemorySystem) -> ToolHandler:
    async def handler(tool_input: dict[str, Any]) -> str:
        try:
            parsed = RememberInput.model_validate(tool_input)
        except ValidationError as exc:
            raise ToolExecutionError(f"参数不合法：{exc}") from exc

        system.remember(parsed.key, parsed.value)
        return json.dumps({"success": True, "key": parsed.key}, ensure_ascii=False)

    return handler


def _make_recall_handler(system: MemorySystem) -> ToolHandler:
    async def handler(tool_input: dict[str, Any]) -> str:
        try:
            parsed = RecallInput.model_validate(tool_input)
        except ValidationError as exc:
            raise ToolExecutionError(f"参数不合法：{exc}") from exc

        hints = system.recall(parsed.query)
        return json.dumps(
            [{"label": hint.label, "text": hint.text} for hint in hints],
            ensure_ascii=False,
        )

    return handler


def register_memory_tools(system: MemorySystem, registry: ToolRegistry) -> None:
    registry.register(
        ToolRegistration(
            definition=ToolDefinition(
                name="remember",
                description=(
                    "记住一条关于主人的长期信息（习惯、说话方式、偏好、性格特点等），不需要"
                    "用户明确要求你才记。key 用稳定的英文路径风格（如 habits/sleep_schedule），"
                    "同一件事尽量复用相同 key 而不是每次编不同的新 key，方便覆盖更新而不是"
                    "产生重复记录。先用 recall 查一下有没有已经记过类似的，避免同一件事存两条。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "稳定的英文路径风格 key，如 habits/sleep_schedule",
                        },
                        "value": {
                            "type": "string",
                            "description": "要记住的具体内容",
                        },
                    },
                    "required": ["key", "value"],
                },
            ),
            handler=_make_remember_handler(system),
        )
    )
    registry.register(
        ToolRegistration(
            definition=ToolDefinition(
                name="recall",
                description="回忆起之前记住的关于主人的信息，按关键词模糊搜索。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词",
                        },
                    },
                    "required": ["query"],
                },
            ),
            handler=_make_recall_handler(system),
        )
    )
