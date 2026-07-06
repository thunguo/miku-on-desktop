"""bridge.py 的回归测试：确认 ``HookEventBus.emit_event`` 能把事件投递给已连接的槽。

写法照抄 ``tests/bridge/test_events.py`` 的 ``_make_bus_with_capture`` 模式——同线程内
直接验证信号投递内容，不实际起后台线程或 Qt 事件循环（跨线程 QueuedConnection 调度是
Qt 自身职责）。
"""

from __future__ import annotations

from miku_on_desk.face.hooks.bridge import HookEventBus
from miku_on_desk.face.hooks.schema import HookEvent


def _make_bus_with_capture() -> tuple[HookEventBus, list[object]]:
    bus = HookEventBus()
    captured: list[object] = []
    bus.hook_event.connect(captured.append)
    return bus, captured


def test_emit_event_delivers_to_connected_slot() -> None:
    bus, captured = _make_bus_with_capture()
    event = HookEvent.from_raw({"event": "SessionStart"})

    bus.emit_event(event)

    assert captured == [event]


def test_emit_event_delivers_multiple_events_in_order() -> None:
    bus, captured = _make_bus_with_capture()
    first = HookEvent.from_raw({"event": "SessionStart"})
    second = HookEvent.from_raw({"event": "SessionEnd"})

    bus.emit_event(first)
    bus.emit_event(second)

    assert captured == [first, second]
