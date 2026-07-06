"""``express_reaction`` 工具：让 LLM 主动驱动桌宠 Miku 的一次性表情/动作反应。

这是 Brain 里少数几个允许依赖 `bridge/` 具体实现的位置之一——`bridge/events.py` 是
Brain 和 face 两侧本来就都依赖的事件桥模块，这里通过 `BrainEventBus.emit_event` 把
`ReactionTriggered` 事件发给 face 侧的 `OverlayWindow`，不引入新的跨层 import。

与 `computer_input`/`screen_analyze` 不同：这个工具没有任何真实世界副作用（不操控鼠标/
键盘，也不读取屏幕），只是往事件总线上发一条通知，所以 `ToolRegistration` 不需要传
`policy_spec=` 覆盖，走默认的 `requires_confirmation=False`。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from miku_on_desk.brain.providers.base import ToolDefinition
from miku_on_desk.brain.tools.registry import (
    ToolExecutionError,
    ToolHandler,
    ToolRegistration,
    ToolRegistry,
)
from miku_on_desk.bridge.events import (
    EXPRESS_REACTION_TOOL_NAME,
    BrainEventBus,
    ReactionKind,
    ReactionTriggered,
)


class ExpressReactionInput(BaseModel):
    kind: ReactionKind


def _make_express_reaction_handler(event_bus: BrainEventBus) -> ToolHandler:
    async def handler(tool_input: dict[str, Any]) -> str:
        try:
            parsed = ExpressReactionInput.model_validate(tool_input)
        except ValidationError as exc:
            raise ToolExecutionError(f"参数不合法：{exc}") from exc

        event_bus.emit_event(ReactionTriggered(kind=parsed.kind))
        return json.dumps({"success": True, "kind": parsed.kind.value}, ensure_ascii=False)

    return handler


def register_express_reaction_tool(event_bus: BrainEventBus, registry: ToolRegistry) -> None:
    registry.register(
        ToolRegistration(
            definition=ToolDefinition(
                name=EXPRESS_REACTION_TOOL_NAME,
                description=(
                    "让桌宠 Miku 做一次表情/动作反应，用来表达你当前对话内容里的情绪或态度。"
                    "不需要每句话都调用，只在情绪明显时用（比如任务顺利完成的喜悦、出错的沮丧、"
                    "被反问/意外内容的惊讶、对新话题的好奇），避免过于频繁而显得聒噪。"
                    "kind 可选：happy（开心/满意）、sad（抱歉/沮丧）、"
                    "surprised（惊讶/意外）、curious（好奇/感兴趣）。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": [k.value for k in ReactionKind],
                            "description": "要表达的反应类型",
                        },
                    },
                    "required": ["kind"],
                },
            ),
            handler=_make_express_reaction_handler(event_bus),
        )
    )
