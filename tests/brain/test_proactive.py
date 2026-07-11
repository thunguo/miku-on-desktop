"""proactive.py 主动交互调度器的回归测试：假 Provider/Backend，不碰真实截图/LLM/OS API。"""

from __future__ import annotations

import asyncio
import queue
from datetime import date, datetime, time
from typing import Any

import pytest

from miku_on_desk.brain.model_router import ModelRouter
from miku_on_desk.brain.proactive import (
    ProactiveToggleRequest,
    ProactiveTrigger,
    _in_quiet_hours,
    _is_quiet_now,
    _next_interval_s,
    _parse_hhmm,
    _peek_and_decide,
    _run_one_iteration,
    apply_proactive_toggle,
)
from miku_on_desk.brain.providers.base import (
    Message,
    OnContent,
    OnThinking,
    Provider,
    StreamResult,
    ToolDefinition,
)
from miku_on_desk.config.settings import (
    ModelRouterConfig,
    ModelTier,
    ProactiveConfig,
    ProviderConfig,
    ProviderName,
)
from miku_on_desk.hands_eyes.backend import ForegroundAppInfo, PlatformBackend, UIElement


class _FakeProvider(Provider):
    """按调用顺序依次返回 ``results`` 里的结果，用完后重复最后一个。"""

    def __init__(self, result: StreamResult | list[StreamResult]) -> None:
        self._results = [result] if isinstance(result, StreamResult) else list(result)
        self.calls: list[dict[str, Any]] = []

    async def stream(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        on_content: OnContent | None = None,
        on_thinking: OnThinking | None = None,
        idle_timeout_s: float = 120.0,
        hard_timeout_s: float = 600.0,
    ) -> StreamResult:
        self.calls.append({"model": model, "messages": list(messages), "tools": tools})
        index = min(len(self.calls) - 1, len(self._results) - 1)
        return self._results[index]


class _FakeBackend(PlatformBackend):
    def __init__(
        self,
        *,
        idle_seconds: float = 0.0,
        app_info: ForegroundAppInfo | None = None,
    ) -> None:
        self._idle_seconds = idle_seconds
        self._app_info = app_info

    def list_elements(self, pid: int) -> list[UIElement]:
        return []

    def get_window_bounds(self, pid: int) -> tuple[int, int, int, int] | None:
        return None

    def open_app(self, name: str) -> None:
        raise NotImplementedError

    def get_idle_seconds(self) -> float:
        return self._idle_seconds

    def get_foreground_app_info(self) -> ForegroundAppInfo | None:
        return self._app_info


def _make_router() -> ModelRouter:
    config = ModelRouterConfig()
    config.anthropic = ProviderConfig(api_key="sk-ant", models={ModelTier.FAST: "claude-fake-fast"})
    return ModelRouter(config)


def test_parse_hhmm_splits_hour_and_minute() -> None:
    assert _parse_hhmm("22:30") == time(hour=22, minute=30)


def test_in_quiet_hours_normal_range_within_same_day() -> None:
    start, end = time(hour=9), time(hour=17)

    assert _in_quiet_hours(time(hour=12), start, end) is True
    assert _in_quiet_hours(time(hour=8), start, end) is False
    assert _in_quiet_hours(time(hour=17), start, end) is False


def test_in_quiet_hours_wraps_past_midnight() -> None:
    start, end = time(hour=22), time(hour=6)

    assert _in_quiet_hours(time(hour=23), start, end) is True
    assert _in_quiet_hours(time(hour=3), start, end) is True
    assert _in_quiet_hours(time(hour=12), start, end) is False


def test_is_quiet_now_returns_false_when_quiet_hours_unset() -> None:
    config = ProactiveConfig()

    assert _is_quiet_now(config, datetime(2026, 7, 5, 23, 0)) is False


def test_is_quiet_now_reflects_configured_range() -> None:
    config = ProactiveConfig(quiet_hours_start="22:00", quiet_hours_end="06:00")

    assert _is_quiet_now(config, datetime(2026, 7, 5, 23, 0)) is True
    assert _is_quiet_now(config, datetime(2026, 7, 5, 12, 0)) is False


def test_next_interval_s_falls_within_configured_range() -> None:
    config = ProactiveConfig(min_interval_s=60, max_interval_s=120)

    for _ in range(20):
        interval = _next_interval_s(config)
        assert 60 <= interval <= 120


async def test_peek_and_decide_returns_trigger_when_should_speak_true() -> None:
    provider = _FakeProvider(
        StreamResult(
            success=True,
            content='{"should_speak": true, "observation": "用户在整理发票"}',
        )
    )
    router = _make_router()
    backend = _FakeBackend(app_info=ForegroundAppInfo(app_name="Excel", window_title="发票.xlsx"))

    trigger = await _peek_and_decide(
        router=router, providers={ProviderName.ANTHROPIC: provider}, backend=backend
    )

    assert trigger == ProactiveTrigger(observation="用户在整理发票")


async def test_peek_and_decide_returns_none_when_should_speak_false() -> None:
    provider = _FakeProvider(
        StreamResult(success=True, content='{"should_speak": false, "observation": ""}')
    )
    router = _make_router()
    backend = _FakeBackend()

    trigger = await _peek_and_decide(
        router=router, providers={ProviderName.ANTHROPIC: provider}, backend=backend
    )

    assert trigger is None


async def test_peek_and_decide_returns_none_on_malformed_json() -> None:
    provider = _FakeProvider(StreamResult(success=True, content="不是 JSON 的文本"))
    router = _make_router()
    backend = _FakeBackend()

    trigger = await _peek_and_decide(
        router=router, providers={ProviderName.ANTHROPIC: provider}, backend=backend
    )

    assert trigger is None


async def test_peek_and_decide_returns_none_on_provider_failure() -> None:
    provider = _FakeProvider(StreamResult(success=False, error="request_timeout"))
    router = _make_router()
    backend = _FakeBackend()

    trigger = await _peek_and_decide(
        router=router, providers={ProviderName.ANTHROPIC: provider}, backend=backend
    )

    assert trigger is None


async def test_peek_and_decide_returns_none_when_observation_missing() -> None:
    provider = _FakeProvider(StreamResult(success=True, content='{"should_speak": true}'))
    router = _make_router()
    backend = _FakeBackend()

    trigger = await _peek_and_decide(
        router=router, providers={ProviderName.ANTHROPIC: provider}, backend=backend
    )

    assert trigger is None


@pytest.fixture
def chat_input() -> queue.Queue[object]:
    return queue.Queue()


async def test_run_one_iteration_skips_when_daily_cap_reached(
    chat_input: queue.Queue[object],
) -> None:
    config = ProactiveConfig(max_daily_triggers=1)
    router = _make_router()
    backend = _FakeBackend()

    daily_count, day_marker = await _run_one_iteration(
        config=config,
        router=router,
        providers={},
        backend=backend,
        chat_input=chat_input,
        daily_count=1,
        day_marker=date(2026, 7, 5),
        now=datetime(2026, 7, 5, 12, 0),
    )

    assert daily_count == 1
    assert day_marker == date(2026, 7, 5)
    assert chat_input.empty()


async def test_run_one_iteration_skips_during_quiet_hours(
    chat_input: queue.Queue[object],
) -> None:
    config = ProactiveConfig(quiet_hours_start="22:00", quiet_hours_end="06:00")
    router = _make_router()
    backend = _FakeBackend()

    daily_count, _day_marker = await _run_one_iteration(
        config=config,
        router=router,
        providers={},
        backend=backend,
        chat_input=chat_input,
        daily_count=0,
        day_marker=date(2026, 7, 5),
        now=datetime(2026, 7, 5, 23, 0),
    )

    assert daily_count == 0
    assert chat_input.empty()


async def test_run_one_iteration_skips_when_user_idle_too_long(
    chat_input: queue.Queue[object],
) -> None:
    config = ProactiveConfig(idle_threshold_s=120)
    router = _make_router()
    backend = _FakeBackend(idle_seconds=300)

    daily_count, _day_marker = await _run_one_iteration(
        config=config,
        router=router,
        providers={},
        backend=backend,
        chat_input=chat_input,
        daily_count=0,
        day_marker=date(2026, 7, 5),
        now=datetime(2026, 7, 5, 12, 0),
    )

    assert daily_count == 0
    assert chat_input.empty()


async def test_run_one_iteration_resets_daily_count_on_new_day(
    chat_input: queue.Queue[object],
) -> None:
    config = ProactiveConfig(max_daily_triggers=1, idle_threshold_s=0)
    router = _make_router()
    backend = _FakeBackend(idle_seconds=999)

    daily_count, day_marker = await _run_one_iteration(
        config=config,
        router=router,
        providers={},
        backend=backend,
        chat_input=chat_input,
        daily_count=1,
        day_marker=date(2026, 7, 4),
        now=datetime(2026, 7, 5, 12, 0),
    )

    assert day_marker == date(2026, 7, 5)
    assert daily_count == 0


async def test_run_one_iteration_skips_and_does_not_count_when_peek_returns_none(
    chat_input: queue.Queue[object],
) -> None:
    provider = _FakeProvider(StreamResult(success=True, content='{"should_speak": false}'))
    config = ProactiveConfig()
    router = _make_router()
    backend = _FakeBackend()

    daily_count, _day_marker = await _run_one_iteration(
        config=config,
        router=router,
        providers={ProviderName.ANTHROPIC: provider},
        backend=backend,
        chat_input=chat_input,
        daily_count=0,
        day_marker=date(2026, 7, 5),
        now=datetime(2026, 7, 5, 12, 0),
    )

    assert daily_count == 0
    assert chat_input.empty()


async def test_run_one_iteration_puts_trigger_and_increments_count_on_success(
    chat_input: queue.Queue[object],
) -> None:
    provider = _FakeProvider(
        StreamResult(
            success=True,
            content='{"should_speak": true, "observation": "用户在看教程视频"}',
        )
    )
    config = ProactiveConfig()
    router = _make_router()
    backend = _FakeBackend()

    daily_count, day_marker = await _run_one_iteration(
        config=config,
        router=router,
        providers={ProviderName.ANTHROPIC: provider},
        backend=backend,
        chat_input=chat_input,
        daily_count=0,
        day_marker=date(2026, 7, 5),
        now=datetime(2026, 7, 5, 12, 0),
    )

    assert daily_count == 1
    assert day_marker == date(2026, 7, 5)
    assert chat_input.get_nowait() == ProactiveTrigger(observation="用户在看教程视频")


async def test_apply_proactive_toggle_starts_scheduler_when_none_running(
    chat_input: queue.Queue[object],
) -> None:
    config = ProactiveConfig()
    router = _make_router()
    backend = _FakeBackend()

    task = await apply_proactive_toggle(
        ProactiveToggleRequest(enabled=True),
        None,
        config=config,
        router=router,
        providers={},
        backend=backend,
        chat_input=chat_input,
    )

    try:
        assert task is not None
        assert not task.done()
    finally:
        assert task is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_apply_proactive_toggle_keeps_existing_task_when_already_running(
    chat_input: queue.Queue[object],
) -> None:
    config = ProactiveConfig()
    router = _make_router()
    backend = _FakeBackend()
    existing_task = await apply_proactive_toggle(
        ProactiveToggleRequest(enabled=True),
        None,
        config=config,
        router=router,
        providers={},
        backend=backend,
        chat_input=chat_input,
    )

    try:
        result = await apply_proactive_toggle(
            ProactiveToggleRequest(enabled=True),
            existing_task,
            config=config,
            router=router,
            providers={},
            backend=backend,
            chat_input=chat_input,
        )

        assert result is existing_task
    finally:
        assert existing_task is not None
        existing_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await existing_task


async def test_apply_proactive_toggle_cancels_running_task_when_disabled(
    chat_input: queue.Queue[object],
) -> None:
    config = ProactiveConfig()
    router = _make_router()
    backend = _FakeBackend()
    running_task = await apply_proactive_toggle(
        ProactiveToggleRequest(enabled=True),
        None,
        config=config,
        router=router,
        providers={},
        backend=backend,
        chat_input=chat_input,
    )

    result = await apply_proactive_toggle(
        ProactiveToggleRequest(enabled=False),
        running_task,
        config=config,
        router=router,
        providers={},
        backend=backend,
        chat_input=chat_input,
    )

    assert result is None
    assert running_task is not None
    assert running_task.cancelled()


async def test_apply_proactive_toggle_is_noop_when_disabled_and_no_task(
    chat_input: queue.Queue[object],
) -> None:
    config = ProactiveConfig()
    router = _make_router()
    backend = _FakeBackend()

    result = await apply_proactive_toggle(
        ProactiveToggleRequest(enabled=False),
        None,
        config=config,
        router=router,
        providers={},
        backend=backend,
        chat_input=chat_input,
    )

    assert result is None
