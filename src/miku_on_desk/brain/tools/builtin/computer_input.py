"""``computer_input`` 工具：把点击/输入文本/按组合键/打开应用这几个真实操控本机的动作，
接到 `hands_eyes.PlatformBackend`。

这是 Brain 里唯一允许依赖 `hands_eyes/` 具体实现的位置——不是对分层规则的破例，而是这层
`builtin/` 工具本身的职责就是"把 Brain 的工具接口接到 hands_eyes 的能力上"，`brain/loop.py`
`brain/tools/registry.py` 等核心模块仍然只认识 `ToolRegistration`/`ToolHandler` 这两个抽象，
从不直接 import `hands_eyes`。

`ToolPolicySpec(requires_confirmation=True)`：这个工具会在用户自己的电脑上产生真实的鼠标/
键盘/进程副作用，执行前必须弹出确认气泡——即使全局 `trusted_mode=True` 也不豁免，这是
`policy.py` 里"requires_confirmation 必须在信任层判断之前生效"规则存在的原因之一。

`PlatformBackend` 的方法都是同步阻塞调用（pynput/psutil/subprocess），必须用
``run_in_executor`` 丢给线程池，否则会卡住 Brain 的 asyncio 事件循环。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from miku_on_desk.brain.providers.base import ToolDefinition
from miku_on_desk.brain.tools.policy import ToolPolicySpec
from miku_on_desk.brain.tools.registry import (
    ToolExecutionError,
    ToolHandler,
    ToolRegistration,
    ToolRegistry,
)
from miku_on_desk.hands_eyes.backend import PlatformBackend

_COMPUTER_INPUT_TOOL_NAME = "computer_input"
_OPEN_APP_PID_RETRIES = 10
_OPEN_APP_PID_RETRY_INTERVAL_S = 0.3
# 找到 PID 后再等一下：open_app 只保证进程存在，不保证窗口已渲染、已拿到焦点——紧跟着的
# type_text/click 如果立刻执行，很容易在应用还没就绪时就发生输入丢失/剪贴板竞态。
_OPEN_APP_SETTLE_DELAY_S = 0.5


class ComputerInputInput(BaseModel):
    action: Literal["click", "type_text", "key_press", "open_app"]
    x: int | None = None
    y: int | None = None
    text: str | None = None
    keys: list[str] | None = None
    app_name: str | None = None


def _find_pid_with_retries(backend: PlatformBackend, name: str) -> int | None:
    for _ in range(_OPEN_APP_PID_RETRIES):
        pid = backend.find_pid_by_name(name)
        if pid is not None:
            return pid
        time.sleep(_OPEN_APP_PID_RETRY_INTERVAL_S)
    return None


def _make_computer_input_handler(backend: PlatformBackend) -> ToolHandler:
    async def handler(tool_input: dict[str, Any]) -> str:
        try:
            parsed = ComputerInputInput.model_validate(tool_input)
        except ValidationError as exc:
            raise ToolExecutionError(f"参数不合法：{exc}") from exc

        loop = asyncio.get_running_loop()
        payload: dict[str, Any] = {"success": True, "action": parsed.action}

        try:
            if parsed.action == "click":
                if parsed.x is None or parsed.y is None:
                    raise ToolExecutionError('action="click" 需要提供 x 和 y')
                await loop.run_in_executor(None, backend.click, parsed.x, parsed.y)
            elif parsed.action == "type_text":
                if parsed.text is None:
                    raise ToolExecutionError('action="type_text" 需要提供 text')
                await loop.run_in_executor(None, backend.type_text, parsed.text)
            elif parsed.action == "key_press":
                if not parsed.keys:
                    raise ToolExecutionError('action="key_press" 需要提供 keys')
                await loop.run_in_executor(None, backend.press_keys, parsed.keys)
            elif parsed.action == "open_app":
                if not parsed.app_name:
                    raise ToolExecutionError('action="open_app" 需要提供 app_name')
                await loop.run_in_executor(None, backend.open_app, parsed.app_name)
                payload["pid"] = await loop.run_in_executor(
                    None, _find_pid_with_retries, backend, parsed.app_name
                )
                if payload["pid"] is not None:
                    await asyncio.sleep(_OPEN_APP_SETTLE_DELAY_S)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"执行操作失败：{exc}") from exc

        return json.dumps(payload, ensure_ascii=False)

    return handler


def register_computer_input_tool(backend: PlatformBackend, registry: ToolRegistry) -> None:
    registry.register(
        ToolRegistration(
            definition=ToolDefinition(
                name=_COMPUTER_INPUT_TOOL_NAME,
                description=(
                    "操控本机鼠标/键盘/应用：click 在指定屏幕坐标单击，type_text 粘贴式输入一段"
                    "文本（支持中文/emoji），key_press 按下一个组合键（如 [\"ctrl\",\"c\"]），"
                    "open_app 启动/唤起一个应用（返回其 pid，可用于 screen_analyze）。"
                    "坐标建议先用 screen_analyze 获取——它返回的 elements 列表里每个元素的"
                    "center_x/center_y 都可直接作为点击坐标使用；只有 vision_grounding 字段"
                    "（找不到文本匹配时的兜底）给出的坐标是估算值，使用前需要更谨慎地确认。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["click", "type_text", "key_press", "open_app"],
                            "description": "要执行的操作类型",
                        },
                        "x": {"type": "integer", "description": "action=click 时的屏幕横坐标"},
                        "y": {"type": "integer", "description": "action=click 时的屏幕纵坐标"},
                        "text": {
                            "type": "string",
                            "description": "action=type_text 时要输入的文本",
                        },
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                'action=key_press 时要按下的组合键，如 ["ctrl", "c"]，'
                                "最后一个元素是主键，其余是修饰键"
                            ),
                        },
                        "app_name": {
                            "type": "string",
                            "description": "action=open_app 时要打开的应用名称",
                        },
                    },
                    "required": ["action"],
                },
            ),
            handler=_make_computer_input_handler(backend),
            policy_spec=ToolPolicySpec(
                requires_confirmation=True,
                confirm_reason="即将操控你的电脑（点击/输入/打开应用），是否允许？",
            ),
        )
    )
