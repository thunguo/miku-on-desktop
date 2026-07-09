"""AI 循环：驱动一次用户输入到最终回复之间的多轮"模型流式响应 ↔ 工具执行"往复。

核心结构是一个有界的工具调用循环：

    while (toolUses.length > 0 && rounds < maxToolRounds) { ... }

每一轮做三件事——按需压缩上下文、按需注入排队消息、追加回合预算软性提醒——然后执行当前
这批工具调用，把结果连同下一轮流式请求的结果一起推进历史，直到模型不再请求任何工具、或
撞上回合预算上限、或 provider 返回错误。

循环里四处不显眼但共同支撑长任务稳定性的机制,按各自依赖的具体子系统是否已存在分成两类：
回合预算软性提醒、墙钟时间软性提醒（``LoopConfig.deadline_s``，供 ``spawn_agents`` 的子
agent 复用，见 ``agents/spawn.py``）都不依赖任何外部子系统，纯靠 rounds 计数/
``time.monotonic()`` 就能实现，直接内置；中途压缩上下文和排队消息注入（本项目目前没有任何
消息队列来源）做成可选的注入点，传 ``None`` 时原样跳过——等对应子系统就位后再传入真正的
回调，不需要改动循环本身。

授权确认这一环做了对应本项目场景的简化：本项目单机单用户，确认回调就是本地一个气泡对话框
的点击结果，不存在"某一路迟迟不响应"的问题，直接 ``await callbacks.confirm(...)`` 就够，不
需要一整套"立即返回 paused、靠独立恢复路径重新进入"的多路并发恢复状态机。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic

from miku_on_desk.brain.model_router import ModelRouter, NoModelAvailableError, ResolvedModel
from miku_on_desk.brain.providers.base import (
    ContentBlock,
    Message,
    OnContent,
    OnThinking,
    Provider,
    StreamResult,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from miku_on_desk.brain.providers.retry import stream_with_retry
from miku_on_desk.brain.tools.policy import Decision
from miku_on_desk.brain.tools.registry import ToolRegistry
from miku_on_desk.brain.tracing import trace_event
from miku_on_desk.config.settings import ModelTier, ProviderName

logger = logging.getLogger(__name__)

ConfirmCallback = Callable[[ToolUseBlock, str | None], Awaitable[bool]]
CompactContextCallback = Callable[[list[Message]], Awaitable[list[Message] | None]]


@dataclass(frozen=True)
class LoopConfig:
    max_tool_rounds: int = 100
    idle_timeout_s: float = 120.0
    hard_timeout_s: float = 600.0
    budget_caution_remaining: int = 10
    budget_critical_remaining: int = 3
    deadline_s: float | None = None
    time_caution_remaining_s: float = 60.0
    time_critical_remaining_s: float = 20.0
    checkin_every_n_rounds: int | None = 8


@dataclass(frozen=True)
class QueuedMessage:
    """长工具链跑到一半时用户插话产生的一条待注入消息。"""

    queued_id: str
    text: str


@dataclass(frozen=True)
class LoopCallbacks:
    """``confirm`` 必填——policy 给出 ASK 决策时必须有地方问用户；其余均为可选的 UI 通知点。"""

    confirm: ConfirmCallback
    on_content: OnContent | None = None
    on_thinking: OnThinking | None = None
    on_tool_use: Callable[[ToolUseBlock], None] | None = None
    on_tool_result: Callable[[ToolResultBlock], None] | None = None
    on_queued_message_injected: Callable[[QueuedMessage], None] | None = None
    consume_queued_message: Callable[[], QueuedMessage | None] | None = None
    compact_context: CompactContextCallback | None = None


class LoopStopReason(StrEnum):
    DONE = "done"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIME_EXHAUSTED = "time_exhausted"
    PROVIDER_ERROR = "provider_error"
    NO_MODEL_AVAILABLE = "no_model_available"
    USER_CANCELLED = "user_cancelled"


@dataclass(frozen=True)
class LoopResult:
    stop_reason: LoopStopReason
    messages: list[Message]
    rounds: int
    error: str | None = None
    raw_error: str | None = None


_PROVIDER_ERROR_DESCRIPTIONS: dict[str, str] = {
    "request_idle_timeout": "模型响应中断，可能是网络问题",
    "request_hard_timeout": "单次请求耗时过长，已强制中断",
}


def _describe_provider_error(error: str | None) -> str | None:
    """把 provider 层的稳定错误 token 转译成面向用户/模型的可操作文案。

    只翻译双超时这两个具体到可操作的 kind；其余 token（``client_error``/
    ``rate_limited`` 等）原样返回——它们已经是稳定的分类结果，硬编码翻译反而会掩盖
    真实原因。未知 token 也原样返回，保持向后兼容。
    """
    if error is None:
        return None
    return _PROVIDER_ERROR_DESCRIPTIONS.get(error, error)


def _assistant_message_from_content(text: str, tool_uses: list[ToolUseBlock]) -> Message:
    blocks: list[ContentBlock] = []
    if text:
        blocks.append(TextBlock(text=text))
    blocks.extend(tool_uses)
    return Message(role="assistant", content=blocks)


def _append_tail_reminder(messages: list[Message], text: str) -> None:
    """把提醒拼进最近一个 tool_result 的尾部，而不是新增一条消息——避免打断 prompt cache 前缀。"""
    for message in reversed(messages):
        if not isinstance(message.content, list):
            continue
        for block in reversed(message.content):
            if isinstance(block, ToolResultBlock):
                block.content = f"{block.content}\n\n{text}"
                return
    logger.debug("没有找到可挂载回合预算提醒的 tool_result，跳过本次提醒：%s", text)


def _budget_warning_text(remaining: int, max_tool_rounds: int, tier: int) -> str:
    if tier == 2:
        return (
            f"[turn-budget] 只剩 {remaining}/{max_tool_rounds} 个工具调用回合。"
            "如果剩余工作做不完，如实告知用户当前进度和还差什么，不要假装已经做完。"
        )
    return (
        f"[turn-budget] 剩余 {remaining}/{max_tool_rounds} 个工具调用回合，"
        "请考虑剩余工作是否能在预算内完成。"
    )


def _time_warning_text(remaining_s: float, tier: int) -> str:
    if tier == 2:
        return (
            f"[time-budget] 剩余时间不足 {remaining_s:.0f} 秒。如果剩余工作做不完，"
            "如实告知用户当前进度和还差什么，不要假装已经做完。"
        )
    return f"[time-budget] 剩余时间约 {remaining_s:.0f} 秒，请考虑剩余工作是否能在预算内完成。"


_VERIFY_BEFORE_DONE_TEXT = (
    "[verify-before-done] 刚才这一步是需要用户确认才能执行的操作。如果接下来打算宣布任务已"
    "完成，先用一次只读检查确认它真的达到了预期效果，不要假设执行成功就等于目标达成。"
)


async def _resolve_tool_call(
    tool_use: ToolUseBlock,
    *,
    registry: ToolRegistry,
    session_id: str,
    callbacks: LoopCallbacks,
    round: int,
) -> ToolResultBlock:
    logger.debug(
        "ai_loop tool_call_start session_id=%s tool=%s tool_use_id=%s",
        session_id,
        tool_use.name,
        tool_use.id,
    )
    trace_event(
        session_id, "tool_call_start", tool=tool_use.name, tool_use_id=tool_use.id, round=round
    )
    if callbacks.on_tool_use is not None:
        callbacks.on_tool_use(tool_use)

    call_start = time.monotonic()
    decision = registry.evaluate(tool_use, session_id=session_id)
    if decision.decision == Decision.DENY:
        result = ToolResultBlock(
            tool_use_id=tool_use.id,
            content=decision.reason or f'工具 "{tool_use.name}" 被拒绝。',
            is_error=True,
        )
    elif decision.decision == Decision.ASK:
        approved = await callbacks.confirm(tool_use, decision.reason)
        if approved:
            result = await registry.execute(tool_use, session_id=session_id)
        else:
            result = ToolResultBlock(
                tool_use_id=tool_use.id, content="用户拒绝了这次操作确认。", is_error=True
            )
    else:
        result = await registry.execute(tool_use, session_id=session_id)

    if not result.is_error:
        policy_spec = registry.policy_spec_for(tool_use.name)
        if policy_spec is not None and policy_spec.requires_confirmation:
            result.content = f"{result.content}\n\n{_VERIFY_BEFORE_DONE_TEXT}"
    duration_ms = (time.monotonic() - call_start) * 1000

    if callbacks.on_tool_result is not None:
        callbacks.on_tool_result(result)
    logger.debug(
        "ai_loop tool_call_end session_id=%s tool=%s tool_use_id=%s is_error=%s",
        session_id,
        tool_use.name,
        tool_use.id,
        result.is_error,
    )
    trace_event(
        session_id,
        "tool_call_end",
        tool=tool_use.name,
        tool_use_id=tool_use.id,
        round=round,
        decision=decision.decision,
        duration_ms=duration_ms,
        is_error=result.is_error,
    )
    return result


async def _stream_with_fallback(
    *,
    session_id: str,
    router: ModelRouter,
    providers: Mapping[ProviderName, Provider],
    resolved: ResolvedModel,
    tier: ModelTier,
    system: str,
    messages: list[Message],
    tools: list[ToolDefinition],
    callbacks: LoopCallbacks,
    config: LoopConfig,
) -> tuple[StreamResult, ResolvedModel]:
    """先对 `resolved` 指向的 Provider 做带重试的流式调用；重试耗尽后若跨 Provider 降级
    开启且能找到替代 Provider，再整体尝试一次替代 Provider。替代 Provider 成功时，返回的
    `ResolvedModel` 会替换调用方持有的那份，让后续轮次继续使用这个证明可用的 Provider，而
    不是每轮都退回原本失败的那个。替代 Provider 也失败时，保留并返回原始 Provider 的错误——
    它是这次失败链条里更有信息量的那一个。
    """
    provider = providers[resolved.provider]
    call_start = time.monotonic()
    result = await stream_with_retry(
        provider,
        model=resolved.model_id,
        system=system,
        messages=messages,
        tools=tools,
        on_content=callbacks.on_content,
        on_thinking=callbacks.on_thinking,
        idle_timeout_s=config.idle_timeout_s,
        hard_timeout_s=config.hard_timeout_s,
    )
    trace_event(
        session_id,
        "provider_call",
        provider=resolved.provider,
        model=resolved.model_id,
        duration_ms=(time.monotonic() - call_start) * 1000,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        cache_creation_input_tokens=result.usage.cache_creation_input_tokens,
        cache_read_input_tokens=result.usage.cache_read_input_tokens,
        stop_reason=result.stop_reason,
        success=result.success,
    )
    if result.success:
        return result, resolved

    fallback = router.resolve_fallback(tier, exclude=resolved.provider)
    if fallback is None:
        return result, resolved

    logger.debug(
        "provider %s 重试耗尽（%s），降级到 %s", resolved.provider, result.error, fallback.provider
    )
    logger.warning(
        "ai_loop fallback_triggered session_id=%s from_provider=%s error=%s to_provider=%s",
        session_id,
        resolved.provider,
        result.error,
        fallback.provider,
    )
    trace_event(
        session_id,
        "fallback_triggered",
        from_provider=resolved.provider,
        to_provider=fallback.provider,
        error=result.error,
    )
    fallback_provider = providers[fallback.provider]
    fallback_call_start = time.monotonic()
    fallback_result = await stream_with_retry(
        fallback_provider,
        model=fallback.model_id,
        system=system,
        messages=messages,
        tools=tools,
        on_content=callbacks.on_content,
        on_thinking=callbacks.on_thinking,
        idle_timeout_s=config.idle_timeout_s,
        hard_timeout_s=config.hard_timeout_s,
    )
    trace_event(
        session_id,
        "provider_call",
        provider=fallback.provider,
        model=fallback.model_id,
        duration_ms=(time.monotonic() - fallback_call_start) * 1000,
        input_tokens=fallback_result.usage.input_tokens,
        output_tokens=fallback_result.usage.output_tokens,
        cache_creation_input_tokens=fallback_result.usage.cache_creation_input_tokens,
        cache_read_input_tokens=fallback_result.usage.cache_read_input_tokens,
        stop_reason=fallback_result.stop_reason,
        success=fallback_result.success,
    )
    if fallback_result.success:
        return fallback_result, fallback
    return result, resolved


async def run_ai_loop(
    *,
    session_id: str,
    tier: ModelTier,
    router: ModelRouter,
    providers: Mapping[ProviderName, Provider],
    registry: ToolRegistry,
    system: str,
    messages: list[Message],
    callbacks: LoopCallbacks,
    config: LoopConfig | None = None,
) -> LoopResult:
    config = config or LoopConfig()
    loop_trace_start = time.monotonic()

    try:
        resolved = router.resolve(tier)
    except NoModelAvailableError as exc:
        logger.warning(
            "ai_loop loop_early_exit session_id=%s stop_reason=%s error=%s",
            session_id,
            LoopStopReason.NO_MODEL_AVAILABLE,
            exc,
        )
        trace_event(
            session_id,
            "loop_early_exit",
            stop_reason=LoopStopReason.NO_MODEL_AVAILABLE,
            error=str(exc),
        )
        return LoopResult(
            stop_reason=LoopStopReason.NO_MODEL_AVAILABLE,
            messages=messages,
            rounds=0,
            error=str(exc),
        )
    working_messages = list(messages)
    logger.info(
        "ai_loop loop_start session_id=%s tier=%s provider=%s model=%s",
        session_id,
        tier,
        resolved.provider,
        resolved.model_id,
    )
    trace_event(
        session_id,
        "loop_start",
        tier=tier,
        provider=resolved.provider,
        model=resolved.model_id,
        max_tool_rounds=config.max_tool_rounds,
        deadline_s=config.deadline_s,
    )
    result, resolved = await _stream_with_fallback(
        session_id=session_id,
        router=router,
        providers=providers,
        resolved=resolved,
        tier=tier,
        system=system,
        messages=working_messages,
        tools=registry.definitions(),
        callbacks=callbacks,
        config=config,
    )
    if not result.success:
        logger.warning(
            "ai_loop loop_early_exit session_id=%s stop_reason=%s error=%s",
            session_id,
            LoopStopReason.PROVIDER_ERROR,
            result.error,
        )
        trace_event(
            session_id,
            "loop_early_exit",
            stop_reason=LoopStopReason.PROVIDER_ERROR,
            error=result.error,
            raw_error=result.raw_error,
        )
        return LoopResult(
            stop_reason=LoopStopReason.PROVIDER_ERROR,
            messages=working_messages,
            rounds=0,
            error=_describe_provider_error(result.error),
            raw_error=result.raw_error,
        )
    working_messages.append(_assistant_message_from_content(result.content, result.tool_uses))
    tool_uses = result.tool_uses

    rounds = 0
    last_budget_warning_tier = 0
    last_time_warning_tier = 0
    start = monotonic()
    time_exhausted = False

    while tool_uses and rounds < config.max_tool_rounds:
        remaining_s = (
            None if config.deadline_s is None else config.deadline_s - (monotonic() - start)
        )
        logger.debug(
            "ai_loop round_start session_id=%s round=%d tool_calls=%d remaining_s=%s",
            session_id,
            rounds,
            len(tool_uses),
            remaining_s,
        )
        if remaining_s is not None and remaining_s <= 0:
            time_exhausted = True
            break

        if rounds > 0 and callbacks.compact_context is not None:
            compacted = await callbacks.compact_context(working_messages)
            if compacted is not None:
                working_messages = compacted

        if callbacks.consume_queued_message is not None:
            queued = callbacks.consume_queued_message()
            if queued is not None:
                working_messages.append(Message(role="user", content=queued.text))
                if callbacks.on_queued_message_injected is not None:
                    callbacks.on_queued_message_injected(queued)

        remaining = config.max_tool_rounds - rounds
        tier_level = 0
        if remaining <= config.budget_critical_remaining:
            tier_level = 2
        elif remaining <= config.budget_caution_remaining:
            tier_level = 1
        if tier_level > last_budget_warning_tier:
            last_budget_warning_tier = tier_level
            warning_text = _budget_warning_text(remaining, config.max_tool_rounds, tier_level)
            _append_tail_reminder(working_messages, warning_text)
            logger.info(
                "ai_loop budget_warning session_id=%s round=%d tier=%d remaining=%d max_rounds=%d",
                session_id,
                rounds,
                tier_level,
                remaining,
                config.max_tool_rounds,
            )
            trace_event(
                session_id,
                "budget_warning",
                round=rounds,
                tier=tier_level,
                remaining=remaining,
            )

        if remaining_s is not None:
            time_tier_level = 0
            if remaining_s <= config.time_critical_remaining_s:
                time_tier_level = 2
            elif remaining_s <= config.time_caution_remaining_s:
                time_tier_level = 1
            if time_tier_level > last_time_warning_tier:
                last_time_warning_tier = time_tier_level
                time_warning_text = _time_warning_text(remaining_s, time_tier_level)
                _append_tail_reminder(working_messages, time_warning_text)
                logger.info(
                    "ai_loop time_warning session_id=%s round=%d tier=%d remaining_s=%.1f",
                    session_id,
                    rounds,
                    time_tier_level,
                    remaining_s,
                )
                trace_event(
                    session_id,
                    "time_warning",
                    round=rounds,
                    tier=time_tier_level,
                    remaining=remaining_s,
                )

        if (
            config.checkin_every_n_rounds is not None
            and rounds > 0
            and rounds % config.checkin_every_n_rounds == 0
        ):
            _append_tail_reminder(
                working_messages,
                f"[progress-checkin] 已经连续执行了 {rounds} 轮工具调用。如果任务范围比预期更大，"
                "或者方向可能跑偏了，先跟用户确认一下现状和下一步该怎么做，不要闷头做完很多轮才"
                "汇报。",
            )
            logger.info("ai_loop progress_checkin session_id=%s round=%d", session_id, rounds)

        tool_results = [
            await _resolve_tool_call(
                tool_use,
                registry=registry,
                session_id=session_id,
                callbacks=callbacks,
                round=rounds,
            )
            for tool_use in tool_uses
        ]
        working_messages.append(Message(role="user", content=list(tool_results)))

        result, resolved = await _stream_with_fallback(
            session_id=session_id,
            router=router,
            providers=providers,
            resolved=resolved,
            tier=tier,
            system=system,
            messages=working_messages,
            tools=registry.definitions(),
            callbacks=callbacks,
            config=config,
        )
        rounds += 1
        if not result.success:
            return LoopResult(
                stop_reason=LoopStopReason.PROVIDER_ERROR,
                messages=working_messages,
                rounds=rounds,
                error=_describe_provider_error(result.error),
                raw_error=result.raw_error,
            )
        working_messages.append(_assistant_message_from_content(result.content, result.tool_uses))
        tool_uses = result.tool_uses

    if time_exhausted:
        stop_reason = LoopStopReason.TIME_EXHAUSTED
    elif tool_uses and rounds >= config.max_tool_rounds:
        stop_reason = LoopStopReason.BUDGET_EXHAUSTED
    else:
        stop_reason = LoopStopReason.DONE
    logger.info(
        "ai_loop loop_end session_id=%s stop_reason=%s rounds=%d",
        session_id,
        stop_reason,
        rounds,
    )
    trace_event(
        session_id,
        "loop_end",
        stop_reason=stop_reason,
        rounds=rounds,
        total_duration_ms=(time.monotonic() - loop_trace_start) * 1000,
    )
    return LoopResult(stop_reason=stop_reason, messages=working_messages, rounds=rounds)
