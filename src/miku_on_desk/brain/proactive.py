"""主动交互调度器：不定时地观察屏幕内容，判断是否有值得主动搭话的时机。

只产出一句"观察摘要"（`ProactiveTrigger.observation`），不直接产出回复文本——真正的回复
仍然通过完整的 `run_ai_loop`（含人格/工具/记忆）生成，保证语气人设与普通对话一致。调度器
决定要说话时直接把 `ProactiveTrigger` 放进 `chat_input` 队列，复用 `main.py` 主循环里
`chat_input.get()` 的既有唤醒机制，不需要新的同步原语。

`_run_one_iteration` 把每次循环体拆成一个显式传入 `now: datetime` 的纯函数，便于测试免打扰
时段判断、每日计数跨天重置等分支，不需要 monkeypatch `datetime` 模块。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import queue
import random
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

from miku_on_desk.brain.model_router import ModelRouter
from miku_on_desk.brain.providers.base import ImageBlock, Message, Provider, TextBlock
from miku_on_desk.config.settings import ModelTier, ProactiveConfig, ProviderName
from miku_on_desk.hands_eyes.backend import PlatformBackend
from miku_on_desk.hands_eyes.capture import capture_screen
from miku_on_desk.hands_eyes.vision_fallback import encode_image_as_base64

_PEEK_TIER = ModelTier.FAST

_PEEK_SYSTEM_PROMPT = (
    "你是 Miku 的观察器：给你一张当前屏幕截图和前台应用信息，判断现在是否有一个自然、"
    "不突兀的时机可以主动跟用户搭句话（给反馈、提醒或者问要不要帮忙）。大多数时候答案"
    "应该是不需要——只有真的看到明确、值得一说的信号时才触发，不要为了触发而找理由。"
    '严格输出 JSON：{"should_speak": true 或 false, "observation": "你注意到的'
    '具体内容，简短一句话"}，不要输出其他文字。should_speak 为 false 时'
    "observation 可以是空字符串。"
)


@dataclass(frozen=True)
class ProactiveTrigger:
    """放入 chat_input 队列的内部信号，text 是一句"观察摘要"，不是最终回复——最终回复
    仍然要走完整 run_ai_loop（含人格/工具/记忆）生成，保证语气人设一致。"""

    observation: str


@dataclass(frozen=True)
class ProactiveToggleRequest:
    """托盘"主动交互"开关放入 chat_input 队列的请求：session 级、不持久化，
    只切换调度器任务的有无，不改变 min_interval_s/quiet_hours 等其余配置字段。"""

    enabled: bool


def _parse_hhmm(value: str) -> time:
    hour_str, minute_str = value.split(":")
    return time(hour=int(hour_str), minute=int(minute_str))


def _in_quiet_hours(now: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= now < end
    return now >= start or now < end  # 跨午夜（如 22:00-06:00）


def _is_quiet_now(config: ProactiveConfig, now: datetime) -> bool:
    if config.quiet_hours_start is None or config.quiet_hours_end is None:
        return False
    return _in_quiet_hours(
        now.time(), _parse_hhmm(config.quiet_hours_start), _parse_hhmm(config.quiet_hours_end)
    )


def _next_interval_s(config: ProactiveConfig) -> float:
    return random.uniform(config.min_interval_s, config.max_interval_s)


def _parse_peek_decision(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


async def _peek_and_decide(
    *,
    router: ModelRouter,
    providers: dict[ProviderName, Provider],
    backend: PlatformBackend,
) -> ProactiveTrigger | None:
    loop = asyncio.get_running_loop()
    screenshot = await loop.run_in_executor(None, capture_screen)
    media_type, data = encode_image_as_base64(screenshot)
    app_info = backend.get_foreground_app_info()
    context_line = (
        f"当前前台应用：{app_info.app_name}，窗口标题：{app_info.window_title}"
        if app_info is not None
        else "无法获取前台应用信息"
    )
    resolved = router.resolve(_PEEK_TIER)
    provider = providers[resolved.provider]
    message = Message(
        role="user",
        content=[
            ImageBlock(media_type=media_type, data=data),
            TextBlock(text=context_line),
        ],
    )
    result = await provider.stream(
        model=resolved.model_id, system=_PEEK_SYSTEM_PROMPT, messages=[message], tools=[]
    )
    if not result.success or not result.content:
        return None
    decision = _parse_peek_decision(result.content)
    if decision is None or not decision.get("should_speak") or not decision.get("observation"):
        return None
    return ProactiveTrigger(observation=decision["observation"])


async def _run_one_iteration(
    *,
    config: ProactiveConfig,
    router: ModelRouter,
    providers: dict[ProviderName, Provider],
    backend: PlatformBackend,
    chat_input: queue.Queue[object],
    daily_count: int,
    day_marker: date,
    now: datetime,
) -> tuple[int, date]:
    if now.date() != day_marker:
        daily_count, day_marker = 0, now.date()
    if daily_count >= config.max_daily_triggers:
        return daily_count, day_marker
    if _is_quiet_now(config, now):
        return daily_count, day_marker
    if backend.get_idle_seconds() >= config.idle_threshold_s:
        return daily_count, day_marker
    trigger = await _peek_and_decide(router=router, providers=providers, backend=backend)
    if trigger is None:
        return daily_count, day_marker
    chat_input.put(trigger)
    return daily_count + 1, day_marker


async def run_proactive_scheduler(
    *,
    config: ProactiveConfig,
    router: ModelRouter,
    providers: dict[ProviderName, Provider],
    backend: PlatformBackend,
    chat_input: queue.Queue[object],
) -> None:
    daily_count = 0
    day_marker = date.today()
    while True:
        await asyncio.sleep(_next_interval_s(config))
        daily_count, day_marker = await _run_one_iteration(
            config=config,
            router=router,
            providers=providers,
            backend=backend,
            chat_input=chat_input,
            daily_count=daily_count,
            day_marker=day_marker,
            now=datetime.now(),
        )


async def apply_proactive_toggle(
    request: ProactiveToggleRequest,
    current_task: asyncio.Task[None] | None,
    *,
    config: ProactiveConfig,
    router: ModelRouter,
    providers: dict[ProviderName, Provider],
    backend: PlatformBackend,
    chat_input: queue.Queue[object],
) -> asyncio.Task[None] | None:
    """把调度器任务调至 ``request.enabled`` 描述的目标状态，幂等——已经在目标状态时不动。"""

    if request.enabled:
        if current_task is not None:
            return current_task
        return asyncio.create_task(
            run_proactive_scheduler(
                config=config,
                router=router,
                providers=providers,
                backend=backend,
                chat_input=chat_input,
            )
        )
    if current_task is None:
        return None
    current_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await current_task
    return None
