"""bridge/events.py 的回归测试：全部在单线程 asyncio 事件循环里验证，不实际起 Qt
主循环或后台线程——跨线程 QueuedConnection 的调度是 Qt 自身职责，这里只锁定"发出的事件
内容对不对""确认闸门的请求/应答能不能配对"这两件事。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from miku_on_desk.brain.loop import LoopCallbacks
from miku_on_desk.brain.memory.system import MemorySystem
from miku_on_desk.brain.model_router import ModelRouter
from miku_on_desk.brain.providers.base import (
    Message,
    OnContent,
    OnThinking,
    Provider,
    StreamResult,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from miku_on_desk.bridge.events import (
    BrainCrashed,
    BrainEventBus,
    BrainRestarting,
    CancellationGate,
    ConfirmationGate,
    ConfirmationRequested,
    ContentDelta,
    QueuedMessageQueue,
    ReactionKind,
    ReactionTriggered,
    ThinkingDelta,
    ToolResultReceived,
    ToolUseStarted,
    build_loop_callbacks,
)
from miku_on_desk.config.settings import ModelRouterConfig, ModelTier, ProviderConfig, ProviderName


def _make_bus_with_capture() -> tuple[BrainEventBus, list[object]]:
    bus = BrainEventBus()
    captured: list[object] = []
    bus.brain_event.connect(captured.append)
    return bus, captured


def _make_router() -> ModelRouter:
    config = ModelRouterConfig()
    config.anthropic = ProviderConfig(
        api_key="sk-ant", models={ModelTier.FAST: "claude-fake-fast"}
    )
    return ModelRouter(config)


@pytest.fixture
def system(tmp_path: Path) -> MemorySystem:
    return MemorySystem(tmp_path / "memory")


class _StubProvider(Provider):
    """这些测试从不真正调用 ``compact_context``，仅用于满足 ``build_loop_callbacks`` 的参数要求。"""

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
        raise NotImplementedError


def _build_loop_callbacks_for_test(
    bus: BrainEventBus,
    gate: ConfirmationGate,
    message_queue: QueuedMessageQueue,
    system: MemorySystem,
) -> LoopCallbacks:
    return build_loop_callbacks(
        bus,
        gate,
        message_queue,
        session_id="s1",
        router=_make_router(),
        providers={ProviderName.ANTHROPIC: _StubProvider()},
        memory_system=system,
    )


def test_build_loop_callbacks_forwards_compaction_tuning_to_make_compact_context(
    monkeypatch: pytest.MonkeyPatch, system: MemorySystem
) -> None:
    captured: dict[str, object] = {}

    def _fake_make_compact_context(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "miku_on_desk.bridge.events.make_compact_context", _fake_make_compact_context
    )

    build_loop_callbacks(
        BrainEventBus(),
        ConfirmationGate(BrainEventBus()),
        QueuedMessageQueue(),
        session_id="s1",
        router=_make_router(),
        providers={ProviderName.ANTHROPIC: _StubProvider()},
        memory_system=system,
        token_threshold=12_345,
        keep_recent=2,
    )

    assert captured["token_threshold"] == 12_345
    assert captured["keep_recent"] == 2


def test_build_loop_callbacks_omits_compaction_tuning_kwargs_when_not_given(
    monkeypatch: pytest.MonkeyPatch, system: MemorySystem
) -> None:
    captured: dict[str, object] = {}

    def _fake_make_compact_context(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "miku_on_desk.bridge.events.make_compact_context", _fake_make_compact_context
    )

    _build_loop_callbacks_for_test(
        BrainEventBus(), ConfirmationGate(BrainEventBus()), QueuedMessageQueue(), system
    )

    assert "token_threshold" not in captured
    assert "keep_recent" not in captured


def test_emit_event_delivers_to_connected_slot() -> None:
    bus, captured = _make_bus_with_capture()

    bus.emit_event(ContentDelta("你好"))

    assert captured == [ContentDelta("你好")]


def test_emit_event_delivers_reaction_triggered() -> None:
    bus, captured = _make_bus_with_capture()

    bus.emit_event(ReactionTriggered(kind=ReactionKind.HAPPY))

    assert captured == [ReactionTriggered(kind=ReactionKind.HAPPY)]


def test_emit_event_delivers_brain_crashed() -> None:
    bus, captured = _make_bus_with_capture()

    bus.emit_event(BrainCrashed(error="炸了"))

    assert captured == [BrainCrashed(error="炸了")]



def test_emit_event_delivers_brain_restarting() -> None:
    bus, captured = _make_bus_with_capture()

    bus.emit_event(BrainRestarting(attempt=1, max_attempts=3, delay_s=1.5, error="炸了"))

    assert captured == [BrainRestarting(attempt=1, max_attempts=3, delay_s=1.5, error="炸了")]


async def test_confirmation_gate_round_trip_returns_approval() -> None:
    bus, captured = _make_bus_with_capture()
    gate = ConfirmationGate(bus)
    tool_use = ToolUseBlock(id="c1", name="computer_input", input={})

    async def responder() -> None:
        await asyncio.sleep(0)
        requested = captured[0]
        assert isinstance(requested, ConfirmationRequested)
        gate.resolve(requested.request_id, True)

    approved, _ = await asyncio.gather(gate.request(tool_use, "why"), responder())

    assert approved is True


async def test_confirmation_gate_round_trip_returns_denial() -> None:
    bus, captured = _make_bus_with_capture()
    gate = ConfirmationGate(bus)
    tool_use = ToolUseBlock(id="c1", name="computer_input", input={})

    async def responder() -> None:
        await asyncio.sleep(0)
        requested = captured[0]
        assert isinstance(requested, ConfirmationRequested)
        gate.resolve(requested.request_id, False)

    approved, _ = await asyncio.gather(gate.request(tool_use, None), responder())

    assert approved is False


async def test_confirmation_gate_cleans_up_pending_after_resolution() -> None:
    bus, captured = _make_bus_with_capture()
    gate = ConfirmationGate(bus)
    tool_use = ToolUseBlock(id="c1", name="computer_input", input={})

    async def responder() -> None:
        await asyncio.sleep(0)
        requested = captured[0]
        assert isinstance(requested, ConfirmationRequested)
        gate.resolve(requested.request_id, True)

    await asyncio.gather(gate.request(tool_use, None), responder())

    assert gate._pending == {}


def test_confirmation_gate_resolve_ignores_unknown_request_id() -> None:
    bus, _ = _make_bus_with_capture()
    gate = ConfirmationGate(bus)

    gate.resolve("does-not-exist", True)


async def test_cancellation_gate_request_stop_cancels_armed_task() -> None:
    gate = CancellationGate()
    task = asyncio.create_task(asyncio.sleep(10))
    gate.arm(task)

    gate.request_stop()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()


def test_cancellation_gate_request_stop_without_armed_task_is_noop() -> None:
    gate = CancellationGate()

    gate.request_stop()


async def test_cancellation_gate_disarm_prevents_further_cancellation() -> None:
    gate = CancellationGate()
    task = asyncio.create_task(asyncio.sleep(10))
    gate.arm(task)
    gate.disarm()

    gate.request_stop()
    await asyncio.sleep(0)

    assert not task.cancelled()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_queued_message_queue_pop_returns_none_when_empty() -> None:
    queue = QueuedMessageQueue()

    assert queue.pop() is None


def test_queued_message_queue_push_then_pop_round_trips() -> None:
    queue = QueuedMessageQueue()

    pushed = queue.push("插一句话")
    popped = queue.pop()

    assert popped == pushed
    assert queue.pop() is None


async def test_build_loop_callbacks_wires_confirm_to_gate_request(system: MemorySystem) -> None:
    bus, captured = _make_bus_with_capture()
    gate = ConfirmationGate(bus)
    callbacks = _build_loop_callbacks_for_test(bus, gate, QueuedMessageQueue(), system)
    tool_use = ToolUseBlock(id="c1", name="computer_input", input={})

    async def responder() -> None:
        await asyncio.sleep(0)
        requested = captured[0]
        assert isinstance(requested, ConfirmationRequested)
        gate.resolve(requested.request_id, True)

    approved, _ = await asyncio.gather(callbacks.confirm(tool_use, "why"), responder())

    assert approved is True


async def test_build_loop_callbacks_wires_content_and_thinking_and_tool_events(
    system: MemorySystem,
) -> None:
    bus, captured = _make_bus_with_capture()
    callbacks = _build_loop_callbacks_for_test(
        bus, ConfirmationGate(bus), QueuedMessageQueue(), system
    )
    tool_use = ToolUseBlock(id="c1", name="computer_input", input={})
    tool_result = ToolResultBlock(tool_use_id="c1", content="ok")

    assert callbacks.on_content is not None
    assert callbacks.on_thinking is not None
    assert callbacks.on_tool_use is not None
    assert callbacks.on_tool_result is not None
    callbacks.on_content("说话内容")
    callbacks.on_thinking("思考过程")
    callbacks.on_tool_use(tool_use)
    callbacks.on_tool_result(tool_result)

    assert captured == [
        ContentDelta("说话内容"),
        ThinkingDelta("思考过程"),
        ToolUseStarted(tool_use),
        ToolResultReceived(tool_result),
    ]


async def test_build_loop_callbacks_wires_consume_queued_message_to_queue_pop(
    system: MemorySystem,
) -> None:
    message_queue = QueuedMessageQueue()
    bus = BrainEventBus()
    callbacks = _build_loop_callbacks_for_test(bus, ConfirmationGate(bus), message_queue, system)
    pushed = message_queue.push("插一句话")

    assert callbacks.consume_queued_message is not None
    assert callbacks.consume_queued_message() == pushed


@pytest.mark.parametrize("field", ["confirm", "on_content", "on_thinking"])
async def test_build_loop_callbacks_sets_required_fields(
    field: str, system: MemorySystem
) -> None:
    bus = BrainEventBus()
    callbacks = _build_loop_callbacks_for_test(
        bus, ConfirmationGate(bus), QueuedMessageQueue(), system
    )

    assert getattr(callbacks, field) is not None


async def test_build_loop_callbacks_sets_non_none_compact_context(system: MemorySystem) -> None:
    bus = BrainEventBus()
    callbacks = _build_loop_callbacks_for_test(
        bus, ConfirmationGate(bus), QueuedMessageQueue(), system
    )

    assert callbacks.compact_context is not None
