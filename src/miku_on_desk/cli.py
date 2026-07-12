"""纯文本 CLI 测试入口：不依赖 Qt GUI/语音，用来在硬件端麦克风/音箱到货前先验证
Brain 对话链路是否正常。用法：``miku -p "发给 Miku 的消息"``。

用 ``QCoreApplication``（不是 ``QApplication``）起一个无 GUI 的 Qt 事件循环——
``BrainEventBus`` 依赖 Qt 的跨线程 QueuedConnection 派发信号，需要有事件循环在跑才能
正确收到；``QCoreApplication`` 不需要 X11/``DISPLAY``，可以直接在纯 SSH 会话里跑，不
依赖屏幕/触屏驱动是否已经装好、校准好。见 ``pyproject.toml`` 的 ``miku`` console script。
"""

from __future__ import annotations

import logging
import sys
from argparse import ArgumentParser

from PySide6.QtCore import QCoreApplication

from miku_on_desk.bridge.events import BrainEvent, LoopFinished
from miku_on_desk.main import (
    _SHUTDOWN,
    _extract_assistant_text,
    load_app_config,
    start_brain_runtime,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(prog="miku")
    parser.add_argument("-p", "--prompt", required=True, help="发给 Miku 的消息")
    args = parser.parse_args(argv)

    config = load_app_config()
    app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    runtime = start_brain_runtime(config)

    reply = ""

    def _on_brain_event(event: BrainEvent) -> None:
        nonlocal reply
        if isinstance(event, LoopFinished):
            reply = _extract_assistant_text(event.result.messages)
            # Brain 线程内部一直在阻塞等下一条消息（``asyncio.to_thread(chat_input.get)``），
            # 一次性 CLI 用完就退，必须显式送这个哨兵值让它退出循环——否则那个阻塞中的线程池
            # 工作线程永远不会完成，Python 解释器退出时的 atexit 清理会永久卡住等它。
            runtime.chat_input.put(_SHUTDOWN)
            app.quit()

    runtime.event_bus.brain_event.connect(_on_brain_event)
    runtime.chat_input.put(args.prompt)

    app.exec()

    runtime.brain_thread.join(timeout=10.0)
    if runtime.brain_thread.is_alive():
        logger.warning("Brain 线程在 10 秒内未能正常退出")

    if not reply:
        print("(没有拿到回复)", file=sys.stderr)
        return 1
    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
