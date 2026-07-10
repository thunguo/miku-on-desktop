"""``spawn_agents`` 工具：并行派发多个子任务给独立跑一段 AI 循环的 sub-agent。

不做"截止时限竞速，超时未完成的任务转入后台继续跑、跑完后通过消息队列异步补投结果"这条路径——
这套背景推进机制在单机单用户项目里没有对应的消息投递通道，硬套只会引入一堆不会被读取的孤儿
状态。这里用更简单的等价物：`asyncio.wait(..., timeout=_DEADLINE_S)`，到点还没完成的任务直接
`cancel()`，作为失败结果报告。

但"直接 cancel 丢弃全部内容"只作为真正卡死场景的兜底——`run_sub_agent` 给 sub-agent 自己的
`run_ai_loop` 传入一个比外层 `asyncio.wait` 提前 `_SUBAGENT_SAFETY_MARGIN_S` 秒到期的
`deadline_s`（见 `loop.py` 的墙钟软限时机制），让 sub-agent 有机会在被外层强制取消前看到
`[time-budget]` 提醒并主动收尾，产出一段真实（哪怕不完整）的回答；只有 sub-agent 自己也没能
在安全边界内收尾、真正跑满整个外层超时的极端情况，才会走到 `cancel()` 丢弃内容这条路。

sub-agent 的确认回调永远自动批准（`_auto_approve`），这只绕过 policy 的 ASK 档——DENY 档（危险
命令正则、路径沙箱越界等结构性拒绝）完全不受影响，见 `loop.py` 里 `resolve_tool_call` 的
DENY/ASK 分支：DENY 从不调用 `confirm`。sub-agent 天生跑在无人值守的后台，没有 UI 可以弹确认框，
所以只有需要人工判断的 ASK 档才适合自动放行。

`spawn_agents` 结构性地把自己排除在每个 sub-agent 能拿到的工具集之外（见 `run_sub_agent` 里
`registry.subset(profile.tools, exclude=(_SPAWN_AGENTS_TOOL_NAME,))`），不依赖内置 profile
恰好没把它写进白名单——`operator` profile 用空 tuple 表示"允许全部已注册工具"，如果不做这层
结构性排除，`operator` sub-agent 会拿到自己派生子 agent 的能力，形成无限递归的风险。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from miku_on_desk.brain.agents.manager import AgentManager
from miku_on_desk.brain.loop import LoopCallbacks, LoopConfig, LoopStopReason, run_ai_loop
from miku_on_desk.brain.model_router import ModelRouter
from miku_on_desk.brain.providers.base import (
    Message,
    Provider,
    TextBlock,
    ToolDefinition,
    ToolUseBlock,
)
from miku_on_desk.brain.tools.registry import (
    ToolExecutionError,
    ToolHandler,
    ToolRegistration,
    ToolRegistry,
)
from miku_on_desk.config.settings import ModelTier, ProviderName

logger = logging.getLogger(__name__)

_SPAWN_AGENTS_TOOL_NAME = "spawn_agents"
_MIN_TASKS = 2
_MAX_TASKS = 20
_DEADLINE_S = 600.0
# sub-agent 自己的软限时安全边界：比外层 asyncio.wait 的截止时限提前这么多秒到期，留出时间让
# loop.py 的 [time-budget] 提醒生效、模型主动收尾，而不是被外层直接 cancel() 丢弃内容。
_SUBAGENT_SAFETY_MARGIN_S = 30.0


@dataclass(frozen=True)
class SubAgentResult:
    id: str
    success: bool
    content: str
    error: str | None
    rounds: int


async def _auto_approve(tool_use: ToolUseBlock, reason: str | None) -> bool:
    return True


def _extract_final_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        if isinstance(message.content, str):
            return message.content
        return "".join(block.text for block in message.content if isinstance(block, TextBlock))
    return ""


async def run_sub_agent(
    *,
    task_id: str,
    task: str,
    agent: str | None,
    tier: ModelTier,
    agent_manager: AgentManager,
    router: ModelRouter,
    providers: Mapping[ProviderName, Provider],
    registry: ToolRegistry,
    host_shell: str,
    deadline_s: float,
) -> SubAgentResult:
    profile = agent_manager.resolve_profile(agent or "researcher")
    if profile is None:
        return SubAgentResult(
            id=task_id, success=False, content="", error=f'未找到 agent profile "{agent}"', rounds=0
        )

    scoped_registry = registry.subset(profile.tools, exclude=(_SPAWN_AGENTS_TOOL_NAME,))
    # sub-agent 没有 main.py 那条按轮次拼 <system-reminder> 的路径，用调用方传入的 host_shell
    # 在这里一次性拼进 system，让 sub-agent 和主循环看到的平台描述保持一致（见 reminder.py 的
    # host_shell_descriptor）。
    system_prompt = f"{profile.system_prompt}\n\n宿主环境：{host_shell}"

    try:
        result = await run_ai_loop(
            session_id=f"subagent:{task_id}",
            tier=tier,
            router=router,
            providers=providers,
            registry=scoped_registry,
            system=system_prompt,
            messages=[Message(role="user", content=task)],
            callbacks=LoopCallbacks(confirm=_auto_approve),
            config=LoopConfig(
                max_tool_rounds=profile.max_rounds,
                deadline_s=max(deadline_s - _SUBAGENT_SAFETY_MARGIN_S, 0.0),
            ),
        )
    except Exception as exc:
        logger.exception('sub-agent 任务 "%s" 执行时抛出未预期异常', task_id)
        return SubAgentResult(id=task_id, success=False, content="", error=str(exc), rounds=0)

    if result.stop_reason in (LoopStopReason.PROVIDER_ERROR, LoopStopReason.NO_MODEL_AVAILABLE):
        return SubAgentResult(
            id=task_id, success=False, content="", error=result.error, rounds=result.rounds
        )

    content = _extract_final_text(result.messages)
    if result.stop_reason == LoopStopReason.TIME_EXHAUSTED:
        return SubAgentResult(
            id=task_id,
            success=True,
            content=content,
            error="时间预算耗尽，已提前收尾",
            rounds=result.rounds,
        )
    success = result.stop_reason == LoopStopReason.DONE
    error = None if success else f"回合预算耗尽（{result.rounds} 轮）"
    return SubAgentResult(
        id=task_id, success=success, content=content, error=error, rounds=result.rounds
    )


class SpawnTaskItem(BaseModel):
    id: str
    task: str
    agent: str | None = None
    model_tier: str | None = None


class SpawnAgentsInput(BaseModel):
    tasks: list[SpawnTaskItem]


def _make_spawn_agents_handler(
    *,
    agent_manager: AgentManager,
    router: ModelRouter,
    providers: Mapping[ProviderName, Provider],
    registry: ToolRegistry,
    host_shell: str,
    deadline_s: float,
) -> ToolHandler:
    async def handler(tool_input: dict[str, Any]) -> str:
        try:
            parsed = SpawnAgentsInput.model_validate(tool_input)
        except ValidationError as exc:
            raise ToolExecutionError(f"参数不合法：{exc}") from exc

        if not (_MIN_TASKS <= len(parsed.tasks) <= _MAX_TASKS):
            raise ToolExecutionError(f"tasks 数量必须在 {_MIN_TASKS}-{_MAX_TASKS} 之间")

        tiers: dict[str, ModelTier] = {}
        for item in parsed.tasks:
            tier_value = item.model_tier or ModelTier.FAST.value
            try:
                tiers[item.id] = ModelTier(tier_value)
            except ValueError as exc:
                raise ToolExecutionError(
                    f'task "{item.id}" 的 model_tier "{tier_value}" 不合法'
                ) from exc

        tasks: dict[str, asyncio.Task[SubAgentResult]] = {
            item.id: asyncio.create_task(
                run_sub_agent(
                    task_id=item.id,
                    task=item.task,
                    agent=item.agent,
                    tier=tiers[item.id],
                    agent_manager=agent_manager,
                    router=router,
                    providers=providers,
                    registry=registry,
                    host_shell=host_shell,
                    deadline_s=deadline_s,
                )
            )
            for item in parsed.tasks
        }

        _done, pending = await asyncio.wait(tasks.values(), timeout=deadline_s)

        results: list[SubAgentResult] = []
        for task_id, task in tasks.items():
            if task in pending:
                task.cancel()
                results.append(
                    SubAgentResult(
                        id=task_id, success=False, content="", error="超时，已取消", rounds=0
                    )
                )
                continue
            try:
                results.append(task.result())
            except Exception as exc:
                results.append(
                    SubAgentResult(id=task_id, success=False, content="", error=str(exc), rounds=0)
                )

        for task in pending:
            with contextlib.suppress(asyncio.CancelledError):
                await task

        payload = {
            "timed_out": bool(pending),
            "results": [
                {
                    "id": r.id,
                    "success": r.success,
                    "content": r.content,
                    "error": r.error,
                    "rounds": r.rounds,
                }
                for r in results
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    return handler


def register_spawn_agents_tool(
    *,
    agent_manager: AgentManager,
    router: ModelRouter,
    providers: Mapping[ProviderName, Provider],
    registry: ToolRegistry,
    host_shell: str,
    deadline_s: float = _DEADLINE_S,
) -> None:
    registry.register(
        ToolRegistration(
            definition=ToolDefinition(
                name=_SPAWN_AGENTS_TOOL_NAME,
                description=(
                    "并行派发多个子任务给独立的 sub-agent 执行，每个子任务可以指定不同的 agent "
                    "profile（researcher/operator/planner 或自定义）和模型层级。子任务之间互不"
                    f"共享上下文，整体超时 {deadline_s:.0f} 秒，超时的子任务会被取消并报告为失败。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "minItems": _MIN_TASKS,
                            "maxItems": _MAX_TASKS,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "string",
                                        "description": "任务标识，用于在结果里对应",
                                    },
                                    "task": {
                                        "type": "string",
                                        "description": "交给 sub-agent 的具体任务描述",
                                    },
                                    "agent": {
                                        "type": "string",
                                        "description": "agent profile 名称，省略则默认 researcher",
                                    },
                                    "model_tier": {
                                        "type": "string",
                                        "enum": [t.value for t in ModelTier],
                                        "description": "模型层级，省略则默认 fast",
                                    },
                                },
                                "required": ["id", "task"],
                            },
                        }
                    },
                    "required": ["tasks"],
                },
            ),
            handler=_make_spawn_agents_handler(
                agent_manager=agent_manager,
                router=router,
                providers=providers,
                registry=registry,
                host_shell=host_shell,
                deadline_s=deadline_s,
            ),
        )
    )
