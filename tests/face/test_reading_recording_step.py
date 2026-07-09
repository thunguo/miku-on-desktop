"""``ReadingRecordingStepWidget`` 的状态机回归测试。

跟 ``test_capture_widgets.py`` 同款手法：真正触碰线程/硬件的私有方法
（``_start_script_generation``/``AudioRecorderWidget.start``）在测试里被整个替换掉，
不需要真的起 ``ReadingScriptWorker`` 线程或触碰麦克风；直接调用状态回调方法模拟
worker/recorder 的终态。
"""

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from miku_on_desk.config.settings import ModelRouterConfig
from miku_on_desk.face.ui import reading_recording_step as step_module
from miku_on_desk.face.ui.reading_recording_step import ReadingRecordingStepWidget


class _FakeReadingScriptWorker(QObject):
    """替身：不起真线程，只提供 ``ReadingScriptWorker`` 的信号/方法形状。"""

    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self, description: str, model_router_config: ModelRouterConfig, parent: Any
    ) -> None:
        del description, model_router_config
        super().__init__(parent)

    def start(self) -> None:
        pass

    def request_cancel(self) -> None:
        pass

    def wait(self, timeout_ms: int) -> None:
        del timeout_ms


def test_start_calls_start_script_generation(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ReadingRecordingStepWidget(ModelRouterConfig(), "一个爱笑的猫娘")
    calls: list[bool] = []
    monkeypatch.setattr(widget, "_start_script_generation", lambda: calls.append(True))

    widget.start()

    assert calls == [True]


def test_on_script_ready_shows_text_and_start_button_without_recording(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ReadingRecordingStepWidget(ModelRouterConfig(), "一个爱笑的猫娘")
    started: list[bool] = []
    monkeypatch.setattr(widget._recorder, "start", lambda: started.append(True))

    widget._on_script_ready("你今天心情怎么样呀")

    assert widget._script_edit.toPlainText() == "你今天心情怎么样呀"
    assert widget._worker is None
    assert not widget._start_recording_button.isHidden()
    assert started == []


def test_start_recording_button_click_hides_button_and_starts_recorder(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ReadingRecordingStepWidget(ModelRouterConfig(), "一个爱笑的猫娘")
    started: list[bool] = []
    monkeypatch.setattr(widget._recorder, "start", lambda: started.append(True))
    widget._on_script_ready("你今天心情怎么样呀")

    widget._start_recording_button.click()

    assert widget._start_recording_button.isHidden()
    assert started == [True]


def test_on_script_failed_shows_error_message(qapp: QApplication) -> None:
    widget = ReadingRecordingStepWidget(ModelRouterConfig(), "一个爱笑的猫娘")

    widget._on_script_failed("没有配置任何 Provider")

    assert widget._worker is None
    assert not widget._error_label.isHidden()
    assert "没有配置任何 Provider" in widget._error_label.text()


def test_on_recording_finished_stores_bytes_and_enables_next_button(qapp: QApplication) -> None:
    widget = ReadingRecordingStepWidget(ModelRouterConfig(), "一个爱笑的猫娘")

    widget._on_recording_finished(b"wav-bytes")

    assert widget._next_button.isEnabled()
    assert widget._next_button.text() == "下一步"


def test_next_button_click_emits_recorded_after_recording_finished(qapp: QApplication) -> None:
    widget = ReadingRecordingStepWidget(ModelRouterConfig(), "一个爱笑的猫娘")
    recorded_calls: list[bytes] = []
    widget.recorded.connect(recorded_calls.append)
    widget._on_recording_finished(b"wav-bytes")

    widget._next_button.click()

    assert recorded_calls == [b"wav-bytes"]


def test_on_recording_unavailable_switches_next_button_to_skip(qapp: QApplication) -> None:
    widget = ReadingRecordingStepWidget(ModelRouterConfig(), "一个爱笑的猫娘")

    widget._on_recording_unavailable("未检测到可用的麦克风")

    assert widget._next_button.isEnabled()
    assert widget._next_button.text() == "跳过声音克隆，仅生成外观"
    assert not widget._error_label.isHidden()
    assert widget._error_label.text() == "未检测到可用的麦克风"


def test_next_button_click_emits_skip_requested_when_recording_unavailable(
    qapp: QApplication,
) -> None:
    widget = ReadingRecordingStepWidget(ModelRouterConfig(), "一个爱笑的猫娘")
    skip_calls: list[None] = []
    widget.skip_requested.connect(lambda: skip_calls.append(None))
    widget._on_recording_unavailable("未检测到可用的麦克风")

    widget._next_button.click()

    assert len(skip_calls) == 1


def test_next_button_click_is_noop_before_any_terminal_state(qapp: QApplication) -> None:
    widget = ReadingRecordingStepWidget(ModelRouterConfig(), "一个爱笑的猫娘")
    recorded_calls: list[bytes] = []
    skip_calls: list[None] = []
    widget.recorded.connect(recorded_calls.append)
    widget.skip_requested.connect(lambda: skip_calls.append(None))

    widget._on_next_clicked()

    assert recorded_calls == []
    assert skip_calls == []


def test_seconds_remaining_changed_updates_countdown_label(qapp: QApplication) -> None:
    widget = ReadingRecordingStepWidget(ModelRouterConfig(), "一个爱笑的猫娘")

    widget._on_seconds_remaining_changed(17)

    assert widget._countdown_label.text() == "录音中… 剩余 17 秒"


def test_retry_clicked_stops_recorder_and_restarts_script_generation(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = ReadingRecordingStepWidget(ModelRouterConfig(), "一个爱笑的猫娘")
    stopped: list[bool] = []
    restarted: list[bool] = []
    monkeypatch.setattr(widget._recorder, "stop", lambda: stopped.append(True))
    monkeypatch.setattr(widget, "_start_script_generation", lambda: restarted.append(True))
    widget._on_recording_finished(b"wav-bytes")

    widget._on_retry_clicked()

    assert stopped == [True]
    assert restarted == [True]


def test_start_script_generation_resets_state_for_retry(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(step_module, "ReadingScriptWorker", _FakeReadingScriptWorker)
    widget = ReadingRecordingStepWidget(ModelRouterConfig(), "一个爱笑的猫娘")
    widget._on_recording_unavailable("未检测到可用的麦克风")

    widget._start_script_generation()

    assert widget._recording_available is True
    assert widget._recorded_bytes is None
    assert widget._error_label.isHidden()
    assert widget._start_recording_button.isHidden()
    assert widget._next_button.text() == "下一步"
    assert not widget._next_button.isEnabled()
    assert widget._worker is not None


def test_shutdown_cancels_pending_worker_and_stops_recorder(qapp: QApplication) -> None:
    widget = ReadingRecordingStepWidget(ModelRouterConfig(), "一个爱笑的猫娘")
    cancel_calls: list[bool] = []
    wait_calls: list[int] = []

    class _FakeWorker:
        def request_cancel(self) -> None:
            cancel_calls.append(True)

        def wait(self, timeout_ms: int) -> None:
            wait_calls.append(timeout_ms)

    fake_worker: Any = _FakeWorker()
    widget._worker = fake_worker
    stopped: list[bool] = []
    widget._recorder.stop = lambda: stopped.append(True)  # type: ignore[method-assign]

    widget.shutdown()

    assert cancel_calls == [True]
    assert wait_calls == [3000]
    assert widget._worker is None
    assert stopped == [True]


def test_shutdown_without_pending_worker_only_stops_recorder(qapp: QApplication) -> None:
    widget = ReadingRecordingStepWidget(ModelRouterConfig(), "一个爱笑的猫娘")
    stopped: list[bool] = []
    widget._recorder.stop = lambda: stopped.append(True)  # type: ignore[method-assign]

    widget.shutdown()

    assert widget._worker is None
    assert stopped == [True]
