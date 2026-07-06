"""HookServer（后台线程）与 UI（Qt 主线程）之间的线程安全事件桥。

结构照抄 ``bridge/events.py`` 的 ``BrainEventBus``：必须在 Qt 主线程构造，其余线程
调用 ``emit_event`` 时，Qt 的 AutoConnection 会自动把回调转发到接收者所在线程的事件
循环里执行，不需要自己维护线程安全队列。
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from miku_on_desk.face.hooks.schema import HookEvent


class HookEventBus(QObject):
    """必须在 UI（Qt 主）线程构造，才能让跨线程 emit 正确落到 QueuedConnection 上。"""

    hook_event = Signal(object)

    def emit_event(self, event: HookEvent) -> None:
        self.hook_event.emit(event)
