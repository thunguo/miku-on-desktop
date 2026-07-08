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

`ComputerUseConfig.enabled` 时额外启用一个结算延迟 + 焦点漂移检测闭环：动作执行后等一小段
时间让 UI 稳定下来，再拿当前前台应用跟"会话目标应用"比对。目标应用是 `_make_computer_input_
handler` 闭包里的一个会话级粘性状态（本项目单进程单会话，闭包生命周期正好等于一次真实会话，
不需要真的接 session_id）：由 open_app 显式设置，此后每次点击/输入/按键都会拿当前前台应用
跟它比对，偏离时持续在 payload 里附带 focus_drift 警告，直到模型用 expect_focus_change=true
确认这次切换、或再次调用 open_app 重新定目标。这是为了防止模型没发现自己已经操作到了错误的
窗口还继续瞎点几轮。默认关闭：涉及"工具执行后自动介入"这一产品行为变化。
"""

from __future__ import annotations

import asyncio
import json
import logging
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
from miku_on_desk.config.settings import ComputerUseConfig
from miku_on_desk.hands_eyes.backend import ForegroundAppInfo, PlatformBackend, element_to_dict

logger = logging.getLogger(__name__)

_COMPUTER_INPUT_TOOL_NAME = "computer_input"
_OPEN_APP_PID_RETRIES = 10
_OPEN_APP_PID_RETRY_INTERVAL_S = 0.3
# 找到 PID 后再等一下：open_app 只保证进程存在，不保证窗口已渲染、已拿到焦点——紧跟着的
# type_text/click 如果立刻执行，很容易在应用还没就绪时就发生输入丢失/剪贴板竞态。
_OPEN_APP_SETTLE_DELAY_S = 0.5

# 短暂出现的系统级提示窗（权限弹窗/UAC 等），不构成模型真正想切换到的目标，出现时既不警告
# 也不重新锁定目标应用。Windows 侧的名字未在真实 Windows 机器上验证过，需要交付前核实——
# 与 `hands_eyes/windows/accessibility.py` 模块文档里同一类风险。
_IGNORED_FOREGROUND_APPS = frozenset({"SecurityAgent", "consent.exe", "LogonUI.exe"})


class ComputerInputInput(BaseModel):
    action: Literal["click", "type_text", "key_press", "open_app"]
    x: int | None = None
    y: int | None = None
    text: str | None = None
    keys: list[str] | None = None
    app_name: str | None = None
    pid: int | None = None
    expect_focus_change: bool = False


def _find_pid_with_retries(backend: PlatformBackend, name: str) -> int | None:
    for _ in range(_OPEN_APP_PID_RETRIES):
        pid = backend.find_pid_by_name(name)
        if pid is not None:
            return pid
        time.sleep(_OPEN_APP_PID_RETRY_INTERVAL_S)
    return None


def _resolve_focus_tracking(
    target: ForegroundAppInfo | None,
    after: ForegroundAppInfo | None,
    *,
    expect_focus_change: bool,
    ignore_apps: frozenset[str],
) -> tuple[ForegroundAppInfo | None, dict[str, Any] | None]:
    """返回 `(下一次要用的目标应用, 若有漂移则为 focus_drift 字段)`。

    纯函数、不做 I/O、不改状态——调用方按返回的第一个元素更新自己的会话级 `target_app`。
    """
    if after is None:
        return target, None
    if target is None:
        return after, None
    if after.app_name == target.app_name:
        return target, None
    if after.app_name in ignore_apps:
        return target, None
    if expect_focus_change:
        return after, None
    drift = {
        "expected_app": target.app_name,
        "actual_app": after.app_name,
        "warning": "检测到焦点已偏离目标应用，注意确认当前操作的窗口是否正确。",
    }
    return target, drift


def _make_computer_input_handler(
    backend: PlatformBackend, computer_use: ComputerUseConfig | None = None
) -> ToolHandler:
    computer_use = computer_use or ComputerUseConfig()
    target_app: ForegroundAppInfo | None = None

    async def handler(tool_input: dict[str, Any]) -> str:
        nonlocal target_app
        try:
            parsed = ComputerInputInput.model_validate(tool_input)
        except ValidationError as exc:
            raise ToolExecutionError(f"参数不合法：{exc}") from exc

        loop = asyncio.get_running_loop()
        payload: dict[str, Any] = {"success": True, "action": parsed.action}
        open_app_pid: int | None = None

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
                open_app_pid = await loop.run_in_executor(
                    None, _find_pid_with_retries, backend, parsed.app_name
                )
                payload["pid"] = open_app_pid
                if open_app_pid is not None:
                    await asyncio.sleep(_OPEN_APP_SETTLE_DELAY_S)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"执行操作失败：{exc}") from exc

        if computer_use.enabled:
            try:
                if parsed.action == "open_app":
                    target_app = await loop.run_in_executor(
                        None, backend.get_foreground_app_info
                    )
                    if open_app_pid is not None:
                        raw_elements = await loop.run_in_executor(
                            None, backend.list_elements, open_app_pid
                        )
                        payload["screen_after"] = {
                            "elements": [element_to_dict(e) for e in raw_elements]
                        }
                else:
                    await asyncio.sleep(computer_use.settle_delay_s)
                    after = await loop.run_in_executor(None, backend.get_foreground_app_info)
                    target_app, drift = _resolve_focus_tracking(
                        target_app,
                        after,
                        expect_focus_change=parsed.expect_focus_change,
                        ignore_apps=_IGNORED_FOREGROUND_APPS,
                    )
                    if drift is not None:
                        payload["focus_drift"] = drift
                    if parsed.pid is not None:
                        raw_elements = await loop.run_in_executor(
                            None, backend.list_elements, parsed.pid
                        )
                        payload["screen_after"] = {
                            "elements": [element_to_dict(e) for e in raw_elements]
                        }
            except Exception as exc:
                logger.warning("Computer Use 闭环（焦点追踪/截图）执行失败：%s", exc)

        return json.dumps(payload, ensure_ascii=False)

    return handler


def register_computer_input_tool(
    backend: PlatformBackend,
    registry: ToolRegistry,
    *,
    computer_use: ComputerUseConfig | None = None,
) -> None:
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
                    "若启用了 Computer Use 闭环：每次操作后会自动比对前台应用与会话内记录的"
                    "目标应用，不一致时在返回结果里附带 focus_drift 警告；操作本来就会切换"
                    "前台应用时（如按下切换应用的快捷键）用 expect_focus_change=true 标记，"
                    "可以抑制这次警告并把新的前台应用重新记为目标；提供 pid 时还会在操作结束后"
                    "附带该窗口的 accessibility 元素快照（screen_after 字段，形状与"
                    "screen_analyze 的 elements 一致）。"
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
                        "pid": {
                            "type": "integer",
                            "description": (
                                "可选，Computer Use 闭环开启时用于在操作结束后附带该进程窗口"
                                "的 accessibility 元素快照（screen_after 字段）"
                            ),
                        },
                        "expect_focus_change": {
                            "type": "boolean",
                            "description": (
                                "可选，默认 false。本次操作预期会切换前台应用时设为 true，"
                                "抑制这次的 focus_drift 警告，并把切换后的应用重新记为会话目标"
                            ),
                        },
                    },
                    "required": ["action"],
                },
            ),
            handler=_make_computer_input_handler(backend, computer_use),
            policy_spec=ToolPolicySpec(
                requires_confirmation=True,
                confirm_reason="即将操控你的电脑（点击/输入/打开应用），是否允许？",
            ),
        )
    )
