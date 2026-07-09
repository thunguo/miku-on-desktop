"""``SpeechController`` 的断句提交、打断状态清理与运行时更换 provider 回归测试。

构造 ``SpeechController`` 时 ``__init__`` 会直接 ``start()`` 一个真正的 ``_SynthWorker``
（``QThread``），本文件统一用 autouse fixture 把 ``_SynthWorker.start`` 换成空实现——
我们只需要验证 UI 线程侧的状态机（提交队列内容、代际计数、worker 是否被换掉），
不依赖真正跑起来的后台线程消费队列、也不需要真的 ``asyncio.run`` 网络合成。
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from miku_on_desk.face.ui.speech_controller import SpeechController, _SynthWorker


class _FakeProvider:
    async def synthesize(self, text: str) -> bytes:
        return b""


@pytest.fixture(autouse=True)
def _no_real_worker_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_SynthWorker, "start", lambda self: None)


def test_feed_submits_completed_sentence_with_current_generation(qapp: QApplication) -> None:
    controller = SpeechController(_FakeProvider())

    controller.feed("你好。")

    generation, text, _ = controller._worker._queue.get_nowait()
    assert generation == controller._generation
    assert text == "你好。"
    controller.close()


def test_flush_submits_buffered_remainder_without_sentence_ending(
    qapp: QApplication,
) -> None:
    controller = SpeechController(_FakeProvider())
    controller.feed("还没说完")

    controller.flush()

    generation, text, _ = controller._worker._queue.get_nowait()
    assert generation == controller._generation
    assert text == "还没说完"
    controller.close()


def test_stop_bumps_generation_and_clears_playback_queue(qapp: QApplication) -> None:
    controller = SpeechController(_FakeProvider())
    initial_generation = controller._generation
    controller._play_queue = [controller._temp_dir / "0.mp3", controller._temp_dir / "1.mp3"]
    controller._current = controller._temp_dir / "2.mp3"

    controller.stop()

    assert controller._generation == initial_generation + 1
    assert controller._play_queue == []
    assert controller._current is None
    assert controller._playing is False
    controller.close()


def test_set_provider_replaces_worker_with_new_instance_bound_to_new_provider(
    qapp: QApplication,
) -> None:
    controller = SpeechController(_FakeProvider())
    old_worker = controller._worker
    new_provider = _FakeProvider()

    controller.set_provider(new_provider)

    assert controller._worker is not old_worker
    assert controller._worker._provider is new_provider
    controller.close()


def test_set_provider_stops_old_worker(qapp: QApplication) -> None:
    controller = SpeechController(_FakeProvider())
    old_worker = controller._worker

    controller.set_provider(_FakeProvider())

    assert old_worker._queue.get_nowait() == _SynthWorker._STOP
    controller.close()


def test_set_provider_bumps_generation_so_stale_audio_is_dropped_after_switch(
    qapp: QApplication,
) -> None:
    controller = SpeechController(_FakeProvider())
    stale_generation = controller._generation

    controller.set_provider(_FakeProvider())
    controller._on_audio_ready(stale_generation, b"stale-audio")

    assert controller._play_queue == []
    controller.close()


def test_close_stops_worker_and_removes_temp_dir(qapp: QApplication) -> None:
    controller = SpeechController(_FakeProvider())
    temp_dir = controller._temp_dir
    assert temp_dir.exists()

    controller.close()

    assert not temp_dir.exists()
