"""``SpeechController`` 的断句提交、流式播放状态机、打断清理与运行时更换 provider 回归测试。

构造 ``SpeechController`` 时 ``__init__`` 会直接 ``start()`` 一个真正的 ``_SynthWorker``
（``QThread``），本文件统一用 autouse fixture 把 ``_SynthWorker.start`` 换成空实现——
我们只需要验证 UI 线程侧的状态机（提交队列内容、代际计数、PCM/mp3 两条播放路径、worker 是否
被换掉），不依赖真正跑起来的后台线程消费队列、也不需要真的 ``asyncio.run`` 网络合成。

假 provider 拆成两种，对应播放侧的两条分流路径：``_FakePcmProvider``（``pcm_format`` 非
``None``，模拟 ElevenLabs 的裸 PCM 输出）与 ``_FakeMp3Provider``（``pcm_format`` 为
``None``，模拟 edge-tts 等只能吐 mp3 分片的引擎）。测试 PCM 播放细节时用 ``_FakeSink``/
``_FakeDevice`` 顶替真实的 ``QAudioSink``，避免依赖真实音频设备与其非确定性时序。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from PySide6.QtWidgets import QApplication

from miku_on_desk.brain.tts.base import PcmFormat
from miku_on_desk.face.ui.speech_controller import SpeechController, _SynthWorker


class _FakeDevice:
    """顶替 ``QAudioSink.start()`` 返回的 ``QIODevice``：只记录写入的字节。"""

    def __init__(self) -> None:
        self.written = bytearray()

    def write(self, data: bytes) -> int:
        self.written.extend(data)
        return len(data)


class _FakeSink:
    """顶替 ``QAudioSink``：记录 ``start``/``stop`` 调用次数，``bytesFree`` 可配置。"""

    def __init__(self, *, bytes_free: int = 1_000_000) -> None:
        self.started = 0
        self.stopped = 0
        self._bytes_free = bytes_free
        self.device = _FakeDevice()

    def start(self) -> _FakeDevice:
        self.started += 1
        return self.device

    def stop(self) -> None:
        self.stopped += 1

    def bytesFree(self) -> int:
        return self._bytes_free


class _FakePcmProvider:
    """模拟 ElevenLabs：裸 PCM 输出，按提交顺序逐块产出 ``chunks``。"""

    def __init__(self, chunks: list[bytes] | None = None) -> None:
        self.pcm_format: PcmFormat | None = PcmFormat(sample_rate=24000)
        self._chunks = chunks if chunks is not None else [b"pcm-bytes"]

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _FakeMp3Provider:
    """模拟 edge-tts/OpenAI 兼容引擎：只能吐压缩容器分片，没有裸 PCM。"""

    def __init__(self, chunks: list[bytes] | None = None) -> None:
        self.pcm_format: PcmFormat | None = None
        self._chunks = chunks if chunks is not None else [b"mp3-bytes"]

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


@pytest.fixture(autouse=True)
def _no_real_worker_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_SynthWorker, "start", lambda self: None)


def test_feed_submits_completed_sentence_with_current_generation(qapp: QApplication) -> None:
    controller = SpeechController(_FakeMp3Provider())

    controller.feed("你好。")

    generation, seq, text = controller._worker._queue.get_nowait()
    assert generation == controller._generation
    assert seq == 0
    assert text == "你好。"
    controller.close()


def test_flush_submits_buffered_remainder_without_sentence_ending(
    qapp: QApplication,
) -> None:
    controller = SpeechController(_FakeMp3Provider())
    controller.feed("还没说完")

    controller.flush()

    generation, seq, text = controller._worker._queue.get_nowait()
    assert generation == controller._generation
    assert seq == 0
    assert text == "还没说完"
    controller.close()


def test_stop_bumps_generation_and_clears_playback_queue(qapp: QApplication) -> None:
    controller = SpeechController(_FakeMp3Provider())
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
    controller = SpeechController(_FakeMp3Provider())
    old_worker = controller._worker
    new_provider = _FakeMp3Provider()

    controller.set_provider(new_provider)

    assert controller._worker is not old_worker
    assert controller._worker._provider is new_provider
    controller.close()


def test_set_provider_stops_old_worker(qapp: QApplication) -> None:
    controller = SpeechController(_FakeMp3Provider())
    old_worker = controller._worker

    controller.set_provider(_FakeMp3Provider())

    assert old_worker._queue.get_nowait() == _SynthWorker._STOP
    controller.close()


def test_set_provider_bumps_generation_so_stale_audio_is_dropped_after_switch(
    qapp: QApplication,
) -> None:
    controller = SpeechController(_FakeMp3Provider())
    stale_generation = controller._generation

    controller.set_provider(_FakeMp3Provider())
    controller._on_audio_chunk_ready(stale_generation, 0, b"stale-audio")

    assert controller._mp3_buffer == bytearray()
    controller.close()


def test_close_stops_worker_and_removes_temp_dir(qapp: QApplication) -> None:
    controller = SpeechController(_FakeMp3Provider())
    temp_dir = controller._temp_dir
    assert temp_dir.exists()

    controller.close()

    assert not temp_dir.exists()


def test_pcm_chunk_reaching_prebuffer_threshold_starts_sink(qapp: QApplication) -> None:
    controller = SpeechController(_FakePcmProvider())
    fake_sink = _FakeSink()
    controller._sink = fake_sink  # type: ignore[assignment]
    threshold = controller._prebuffer_threshold_bytes
    chunk = b"\x00" * threshold

    controller._on_audio_chunk_ready(controller._generation, 0, chunk)

    assert fake_sink.started == 1
    assert controller._sink_started is True
    assert bytes(fake_sink.device.written) == chunk
    controller.close()


def test_pcm_chunk_below_threshold_does_not_start_sink_yet(qapp: QApplication) -> None:
    controller = SpeechController(_FakePcmProvider())
    fake_sink = _FakeSink()
    controller._sink = fake_sink  # type: ignore[assignment]
    short_chunk = b"\x00" * 10

    controller._on_audio_chunk_ready(controller._generation, 0, short_chunk)

    assert fake_sink.started == 0
    assert controller._sink_started is False
    assert controller._backlog == bytearray(short_chunk)
    controller.close()


def test_short_sentence_done_forces_pcm_flush_below_threshold(qapp: QApplication) -> None:
    controller = SpeechController(_FakePcmProvider())
    fake_sink = _FakeSink()
    controller._sink = fake_sink  # type: ignore[assignment]
    short_chunk = b"\x00" * 10
    controller._on_audio_chunk_ready(controller._generation, 0, short_chunk)
    assert controller._sink_started is False

    controller._on_sentence_done(controller._generation, 0)

    assert controller._sink_started is True
    assert fake_sink.started == 1
    assert bytes(fake_sink.device.written) == short_chunk
    controller.close()


def test_mp3_chunks_accumulate_and_flush_on_sentence_done(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = SpeechController(_FakeMp3Provider())
    pump_calls: list[bool] = []
    monkeypatch.setattr(controller, "_pump", lambda: pump_calls.append(True))
    generation = controller._generation

    controller._on_audio_chunk_ready(generation, 0, b"mp3-part-1")
    controller._on_audio_chunk_ready(generation, 0, b"mp3-part-2")
    assert controller._mp3_buffer == bytearray(b"mp3-part-1mp3-part-2")

    controller._on_sentence_done(generation, 0)

    assert controller._mp3_buffer == bytearray()
    assert len(controller._play_queue) == 1
    assert controller._play_queue[0].read_bytes() == b"mp3-part-1mp3-part-2"
    assert pump_calls == [True]
    controller.close()


def test_stop_clears_pcm_backlog_and_stops_sink_and_drops_stale_generation(
    qapp: QApplication,
) -> None:
    controller = SpeechController(_FakePcmProvider())
    fake_sink = _FakeSink()
    controller._sink = fake_sink  # type: ignore[assignment]
    controller._backlog.extend(b"\x00" * 100)
    controller._sink_started = True
    controller._drip_timer.start(25)
    stale_generation = controller._generation

    controller.stop()

    assert fake_sink.stopped == 1
    assert controller._backlog == bytearray()
    assert controller._sink_started is False
    assert controller._drip_timer.isActive() is False

    controller._on_audio_chunk_ready(stale_generation, 0, b"\x00" * 10_000)
    controller._on_sentence_done(stale_generation, 0)

    assert controller._backlog == bytearray()
    assert controller._sink_started is False
    controller.close()


def test_set_provider_rebuilds_sink_when_pcm_format_changes(qapp: QApplication) -> None:
    controller = SpeechController(_FakePcmProvider())
    fake_sink = _FakeSink()
    controller._sink = fake_sink  # type: ignore[assignment]

    controller.set_provider(_FakeMp3Provider())

    # 一次来自 set_provider() 打断时的 stop()，一次来自格式变化时 _configure_for_provider()
    # 主动重建 sink。
    assert fake_sink.stopped == 2
    assert controller._sink is None
    assert controller._pcm_format is None
    controller.close()


def test_set_provider_keeps_sink_when_pcm_format_unchanged(qapp: QApplication) -> None:
    controller = SpeechController(_FakePcmProvider())
    fake_sink = _FakeSink()
    controller._sink = fake_sink  # type: ignore[assignment]

    controller.set_provider(_FakePcmProvider())

    # 格式没变，_configure_for_provider() 不会再多停一次或重建；唯一的 stop() 来自
    # set_provider() 打断时的常规清理。
    assert controller._sink is fake_sink
    assert fake_sink.stopped == 1
    controller.close()
