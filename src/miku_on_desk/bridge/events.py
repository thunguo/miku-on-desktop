"""Brain（后台 asyncio 线程）与 UI（Qt 主线程）之间的线程安全事件桥。

Brain 的 AI 循环跑在独立的 asyncio 事件循环线程上，UI 是 Qt 主线程；两者不能共享可变
状态，只能通过消息传递通信。Brain→UI 方向天然是广播通知，用 Qt 信号最省事：
``BrainEventBus`` 是一个应当在 UI 线程构造的 QObject，只要连接用默认的
AutoConnection，Qt 会在信号发出线程与接收线程不同时自动切换成 QueuedConnection，把
回调转发到接收者所在线程的事件循环里执行——这意味着 Brain 线程可以直接调用
``bus.emit_event(...)``，不需要自己维护线程安全队列。

UI→Brain 方向反过来是"等待一个具体回复"：确认框的是/否结果必须送回正在 await 的那个
协程。``asyncio.Future`` 本身不是线程安全的，必须通过 ``loop.call_soon_threadsafe``
把"设置结果"这个操作转扔回 Brain 所在的事件循环线程去执行，``ConfirmationGate`` 就是
这层包装。排队消息（用户在长工具链跑到一半时插话）不需要等回复，只需要一个线程安全的
先进先出队列，用标准库 ``queue.Queue`` 即可，不需要再包一层 asyncio 机制。
"""

from __future__ import annotations

import asyncio
import queue
import uuid
from dataclasses import dataclass
from enum import StrEnum

from PySide6.QtCore import QObject, Signal

from miku_on_desk.brain.loop import LoopCallbacks, LoopResult, QueuedMessage
from miku_on_desk.brain.memory.compaction import make_compact_context
from miku_on_desk.brain.memory.system import MemorySystem
from miku_on_desk.brain.model_router import ModelRouter
from miku_on_desk.brain.providers.base import Provider, ToolResultBlock, ToolUseBlock
from miku_on_desk.config.settings import ProviderName

EXPRESS_REACTION_TOOL_NAME = "express_reaction"


@dataclass(frozen=True)
class ContentDelta:
    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    text: str


@dataclass(frozen=True)
class AcpChunkReceived:
    """`acp_delegate` 委派期间外部 agent 流式吐出的中间文本片段，区别于 Miku 自己的
    `ContentDelta`——UI 侧需要用不同的气泡样式标出"这是外部 agent 在说话"。
    """

    agent: str
    text: str


@dataclass(frozen=True)
class ToolUseStarted:
    tool_use: ToolUseBlock


@dataclass(frozen=True)
class ToolResultReceived:
    result: ToolResultBlock


@dataclass(frozen=True)
class ConfirmationRequested:
    request_id: str
    tool_use: ToolUseBlock
    reason: str | None


@dataclass(frozen=True)
class QueuedMessageInjected:
    queued: QueuedMessage


@dataclass(frozen=True)
class LoopFinished:
    result: LoopResult


class ReactionKind(StrEnum):
    """``express_reaction`` 工具可选的反应类型，1:1 映射到 4 个瞬态兼容的 ``PetState``。"""

    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    CURIOUS = "curious"


@dataclass(frozen=True)
class ReactionTriggered:
    kind: ReactionKind


BrainEvent = (
    ContentDelta
    | ThinkingDelta
    | AcpChunkReceived
    | ToolUseStarted
    | ToolResultReceived
    | ConfirmationRequested
    | QueuedMessageInjected
    | LoopFinished
    | ReactionTriggered
)


class BrainEventBus(QObject):
    """必须在 UI（Qt 主）线程构造，才能让跨线程 emit 正确落到 QueuedConnection 上。"""

    brain_event = Signal(object)

    def emit_event(self, event: BrainEvent) -> None:
        self.brain_event.emit(event)


def _set_future_result(future: asyncio.Future[bool], value: bool) -> None:
    if not future.done():
        future.set_result(value)


class ConfirmationGate:
    """把 policy 的 ASK 决策转成一次跨线程往返。

    Brain 线程调用 ``request`` 发出确认请求并挂起等待；UI 线程收到
    :class:`ConfirmationRequested` 事件、拿到用户点击结果后调用 ``resolve`` 把结果
    送回去。
    """

    def __init__(self, bus: BrainEventBus) -> None:
        self._bus = bus
        self._pending: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Future[bool]]] = {}

    async def request(self, tool_use: ToolUseBlock, reason: str | None) -> bool:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        request_id = uuid.uuid4().hex
        self._pending[request_id] = (loop, future)
        try:
            self._bus.emit_event(ConfirmationRequested(request_id, tool_use, reason))
            return await future
        finally:
            self._pending.pop(request_id, None)

    def resolve(self, request_id: str, approved: bool) -> None:
        """从 UI 线程调用；request_id 未知（如已被清理）时静默忽略。"""
        pending = self._pending.get(request_id)
        if pending is None:
            return
        loop, future = pending
        loop.call_soon_threadsafe(_set_future_result, future, approved)


class CancellationGate:
    """UI 线程点击"停止"按钮时，用于取消 Brain 线程当前正在跑的那个 ``run_ai_loop`` Task。

    比 ``ConfirmationGate`` 简单：不需要等待一个返回值——``task.cancel()`` 本身就是
    "发出取消请求"这个单向操作的全部，真正的取消结果（``asyncio.CancelledError``
    从 ``await`` 点向上传播）由 ``main.py`` 的 ``_brain_main`` 直接捕获处理，不经过
    这个网关。
    """

    def __init__(self) -> None:
        self._armed: tuple[asyncio.AbstractEventLoop, asyncio.Task[LoopResult]] | None = None

    def arm(self, task: asyncio.Task[LoopResult]) -> None:
        self._armed = (asyncio.get_running_loop(), task)

    def disarm(self) -> None:
        self._armed = None

    def request_stop(self) -> None:
        """UI 线程调用；当前没有正在跑的任务时静默忽略。"""
        if self._armed is None:
            return
        loop, task = self._armed
        loop.call_soon_threadsafe(task.cancel)


class QueuedMessageQueue:
    """UI 线程 push，Brain 线程（loop.py 的 ``consume_queued_message`` 回调）非阻塞 pop。"""

    def __init__(self) -> None:
        self._queue: queue.Queue[QueuedMessage] = queue.Queue()

    def push(self, text: str) -> QueuedMessage:
        queued = QueuedMessage(queued_id=uuid.uuid4().hex, text=text)
        self._queue.put(queued)
        return queued

    def pop(self) -> QueuedMessage | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


def build_loop_callbacks(
    bus: BrainEventBus,
    confirm_gate: ConfirmationGate,
    message_queue: QueuedMessageQueue,
    *,
    session_id: str,
    router: ModelRouter,
    providers: dict[ProviderName, Provider],
    memory_system: MemorySystem,
) -> LoopCallbacks:
    """把 loop.py 的回调协议整体接到事件总线 + 确认闸门 + 排队消息队列上。"""
    return LoopCallbacks(
        confirm=confirm_gate.request,
        on_content=lambda text: bus.emit_event(ContentDelta(text)),
        on_thinking=lambda text: bus.emit_event(ThinkingDelta(text)),
        on_tool_use=lambda tool_use: bus.emit_event(ToolUseStarted(tool_use)),
        on_tool_result=lambda result: bus.emit_event(ToolResultReceived(result)),
        on_queued_message_injected=lambda queued: bus.emit_event(QueuedMessageInjected(queued)),
        consume_queued_message=message_queue.pop,
        compact_context=make_compact_context(
            session_id=session_id,
            router=router,
            providers=providers,
            memory_system=memory_system,
        ),
    )
