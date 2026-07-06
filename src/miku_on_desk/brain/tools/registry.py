"""工具注册表：登记工具定义/处理函数/策略画像，并驱动 evaluate→execute 两阶段执行。

两阶段拆开是为了让"等待用户确认"这件需要跨越 AI 循环和 UI 的异步操作，与"执行工具、拿到结果"
这件纯 Brain 内部操作解耦：``loop.py`` 只需要在 ``evaluate`` 拿到 ``ASK`` 决策后，通过它自己
持有的回调去问 UI,registry 完全不需要知道确认长什么样、走哪条 UI 通道。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miku_on_desk.brain.providers.base import ToolDefinition, ToolResultBlock, ToolUseBlock
from miku_on_desk.brain.tools.policy import Decision, PolicyDecision, PolicyEngine, ToolPolicySpec
from miku_on_desk.brain.tools.read_tracker import ReadTracker

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


class ToolExecutionError(Exception):
    """处理函数主动抛出的已知失败原因；registry 会把它转成 ``is_error`` 的 ``ToolResultBlock``。"""


@dataclass(frozen=True)
class ToolRegistration:
    definition: ToolDefinition
    handler: ToolHandler
    policy_spec: ToolPolicySpec = field(default_factory=ToolPolicySpec)
    marks_read: bool = False
    """成功执行后是否要给 ``ReadTracker`` 打标记，供后续 write 类工具的先读后改检查使用。"""


class ToolRegistry:
    def __init__(self, policy: PolicyEngine, read_tracker: ReadTracker) -> None:
        self._policy = policy
        self._read_tracker = read_tracker
        self._tools: dict[str, ToolRegistration] = {}

    def register(self, registration: ToolRegistration) -> None:
        self._tools[registration.definition.name] = registration

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def definitions(self) -> list[ToolDefinition]:
        return [reg.definition for reg in self._tools.values()]

    def subset(self, names: Iterable[str], *, exclude: Iterable[str] = ()) -> ToolRegistry:
        """返回一个共享同一个 policy/read_tracker 的新 registry，只暴露 names 里列出的工具；
        names 为空表示"当前已注册的全部工具"（对应子 agent 白名单的"允许全部"语义）。
        exclude 里的名字无条件排除，即使出现在 names 中或 names 为空——用于结构性禁止某些工具
        （例如子 agent 永远不能再拿到 spawn_agents，无论 profile 白名单怎么写）。
        """
        allow = set(names)
        deny = set(exclude)
        scoped = ToolRegistry(self._policy, self._read_tracker)
        for tool_name, registration in self._tools.items():
            if tool_name in deny:
                continue
            if allow and tool_name not in allow:
                continue
            scoped._tools[tool_name] = registration
        return scoped

    def evaluate(self, tool_use: ToolUseBlock, *, session_id: str) -> PolicyDecision:
        registration = self._tools.get(tool_use.name)
        if registration is None:
            return PolicyDecision(Decision.DENY, f'未知工具 "{tool_use.name}"。')
        return self._policy.evaluate(
            tool_use.name, tool_use.input, registration.policy_spec, session_id=session_id
        )

    async def execute(self, tool_use: ToolUseBlock, *, session_id: str) -> ToolResultBlock:
        registration = self._tools.get(tool_use.name)
        if registration is None:
            return ToolResultBlock(
                tool_use_id=tool_use.id, content=f'未知工具 "{tool_use.name}"。', is_error=True
            )
        try:
            content = await registration.handler(tool_use.input)
        except ToolExecutionError as exc:
            return ToolResultBlock(tool_use_id=tool_use.id, content=str(exc), is_error=True)
        except Exception:
            logger.exception('工具 "%s" 执行时抛出未预期的异常', tool_use.name)
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=f'工具 "{tool_use.name}" 执行失败（内部错误）。',
                is_error=True,
            )
        if registration.marks_read and registration.policy_spec.path_arg is not None:
            raw_path = tool_use.input.get(registration.policy_spec.path_arg)
            if isinstance(raw_path, str):
                self._read_tracker.mark_read(session_id, Path(raw_path))
        return ToolResultBlock(tool_use_id=tool_use.id, content=content)
