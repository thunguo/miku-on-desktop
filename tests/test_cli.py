"""``cli.py`` 的回归测试：`main()` 把 `-p` 传入的文本塞进 Brain 的 `chat_input`
队列，收到 `LoopFinished` 事件后打印最终回复并以 0 退出；拿不到回复时以非 0 退出，
不往 stdout 写任何内容。真实 `load_app_config`/`start_brain_runtime` 依赖完整配置
和一个真正跑 `run_ai_loop` 的 Brain 线程，这里用一个只回一句固定回复的假 Brain 线程
替身，不重新验证 `_brain_main` 本身（那是 `test_main.py` 的职责）。
"""

from __future__ import annotations

import queue
import threading
from unittest.mock import Mock, patch

import pytest
from PySide6.QtWidgets import QApplication

from miku_on_desk.brain.loop import LoopResult, LoopStopReason
from miku_on_desk.brain.providers.base import Message
from miku_on_desk.bridge.events import BrainEventBus, LoopFinished
from miku_on_desk.cli import main
from miku_on_desk.main import BrainRuntime


def _fake_start_brain_runtime(reply_messages: list[Message]) -> BrainRuntime:
    event_bus = BrainEventBus()
    chat_input: queue.Queue[object] = queue.Queue()

    def _worker() -> None:
        chat_input.get()
        result = LoopResult(stop_reason=LoopStopReason.DONE, messages=reply_messages, rounds=1)
        event_bus.emit_event(LoopFinished(result))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    return BrainRuntime(
        event_bus=event_bus,
        confirm_gate=Mock(),
        cancellation_gate=Mock(),
        message_queue=Mock(),
        chat_input=chat_input,
        session_id="test-session",
        memory_system=Mock(),
        brain_thread=thread,
    )


def test_main_prints_final_assistant_reply_and_returns_zero(
    qapp: QApplication, capsys: pytest.CaptureFixture[str]
) -> None:
    messages = [Message(role="assistant", content="你好呀")]
    with (
        patch("miku_on_desk.cli.load_app_config", return_value=Mock()),
        patch(
            "miku_on_desk.cli.start_brain_runtime",
            return_value=_fake_start_brain_runtime(messages),
        ),
    ):
        exit_code = main(["-p", "你好"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "你好呀"


def test_main_returns_nonzero_and_writes_nothing_to_stdout_when_no_assistant_reply(
    qapp: QApplication, capsys: pytest.CaptureFixture[str]
) -> None:
    with (
        patch("miku_on_desk.cli.load_app_config", return_value=Mock()),
        patch(
            "miku_on_desk.cli.start_brain_runtime",
            return_value=_fake_start_brain_runtime([]),
        ),
    ):
        exit_code = main(["-p", "你好"])

    assert exit_code == 1
    assert capsys.readouterr().out == ""
