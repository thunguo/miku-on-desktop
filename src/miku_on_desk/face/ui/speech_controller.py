"""把 Miku 的流式回复文本变成实时播放的语音。

三段职责，都汇聚在 UI（Qt 主）线程上的 :class:`SpeechController`：

1. **断句**——``feed`` 把 ``ContentDelta`` 增量喂给 :class:`SentenceBuffer`，凑成整句才提交，
   避免逐字合成造成的破碎语音；``flush`` 在一轮回复结束时补上最后没有标点收尾的残句。
2. **合成**——句子提交给后台 :class:`_SynthWorker` 线程（TTS 是网络 IO，不能阻塞 UI 线程）。
   worker 单线程 FIFO 逐句合成，因此音频 chunk 天然按提交顺序返回，播放顺序无需额外排序；
   worker 不再等整句音频攒完才回传，而是逐 chunk 转发，句内首字节即可开始播放。
3. **播放**——按 provider 的 ``pcm_format`` 分两条路径：

   - **裸 PCM**（如 ElevenLabs）：chunk 直接进入一个跨句子连续的 ``self._backlog``，
     由 ``self._sink``（``QAudioSink``，push 模式）+ ``self._drip_timer`` 定期抽取写入，
     句子边界对播放层透明——上一句的尾音和下一句的首音在同一个字节流里前后相接，
     不需要等一句放完再等下一句。首次出声前有一个很短的冷启动预缓冲，避免起播爆音/卡顿。
   - **压缩容器**（如 mp3，edge-tts 协议限制无法提供 PCM）：仍然要整句 chunk 攒完才能交给
     ``QMediaPlayer`` 解码播放（分片的 mp3 帧不能直接拼接播放），退回原来的临时文件 +
     顺序播放队列逻辑，只是触发时机从"一次拿到完整 bytes"变成"多个 chunk 累积 +
     该句合成完成通知"。

``stop`` 用一个"代际"计数器丢弃所有在途/排队的音频：打断时递增代际，之前提交的句子即便合成
完成、其携带的旧代际也会在回调里被识别为过期而丢弃，不会串到下一轮回复里；PCM 播放侧额外需要
主动停止 ``QAudioSink`` 并清空积压字节，因为这条路径下音频不再是"放完一个文件"就自然截止。
"""

from __future__ import annotations

import asyncio
import logging
import queue
import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import QIODevice, QObject, QThread, QTimer, QUrl, Signal
from PySide6.QtMultimedia import (
    QAudioFormat,
    QAudioOutput,
    QAudioSink,
    QMediaPlayer,
)

from miku_on_desk.brain.tts.base import PcmFormat, TTSProvider
from miku_on_desk.brain.tts.sentence_buffer import SentenceBuffer
from miku_on_desk.brain.tts.text_sanitizer import sanitize_for_speech

logger = logging.getLogger(__name__)

_DRIP_INTERVAL_MS = 25
_PREBUFFER_SECONDS = 0.15
_BYTES_PER_SAMPLE = 2  # 16-bit PCM


class _SynthWorker(QThread):
    """后台合成线程：从队列取句子，调用 provider 合成，逐 chunk 把音频字节发回 UI 线程。

    ``audio_chunk_ready``/``sentence_done`` 从 ``run()`` 所在的 worker 线程发出，经 Qt 默认的
    AutoConnection 排队投递到 UI 线程的槽——这正是让网络合成不阻塞 UI、又不必手写线程同步的
    关键；同一线程发出的信号保证按发出顺序进入 UI 线程队列，因此不需要额外的乱序处理。
    """

    audio_chunk_ready = Signal(int, int, bytes)  # (generation, seq, chunk)
    sentence_done = Signal(int, int)  # (generation, seq)

    _STOP = (None, None, None)

    def __init__(self, provider: TTSProvider, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._provider = provider
        self._queue: queue.Queue[tuple[int | None, int | None, str | None]] = queue.Queue()

    def submit(self, generation: int, seq: int, text: str) -> None:
        self._queue.put((generation, seq, text))

    def stop(self) -> None:
        self._queue.put(self._STOP)

    def run(self) -> None:
        while True:
            generation, seq, text = self._queue.get()
            if generation is None or seq is None or text is None:
                return
            asyncio.run(self._synthesize_one(generation, seq, text))

    async def _synthesize_one(self, generation: int, seq: int, text: str) -> None:
        try:
            async for chunk in self._provider.synthesize_stream(text):
                if chunk:
                    self.audio_chunk_ready.emit(generation, seq, chunk)
        except Exception:
            logger.exception("TTS 流式合成失败，跳过该句：%r", text)
        finally:
            self.sentence_done.emit(generation, seq)


class SpeechController(QObject):
    """UI 线程上的语音控制器：喂文本进来，它负责断句、后台合成、实时播放。"""

    def __init__(self, provider: TTSProvider, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._buffer = SentenceBuffer()
        self._generation = 0
        self._next_seq = 0
        self._pending_sentences = 0

        # PCM 播放路径的状态；_pcm_format 为 None 时表示当前 provider 不支持裸 PCM，
        # 走下面的 mp3 兜底路径。
        self._pcm_format: PcmFormat | None = None
        self._prebuffer_threshold_bytes = 0
        self._sink: QAudioSink | None = None
        self._sink_device: QIODevice | None = None
        self._sink_started = False
        self._backlog = bytearray()
        self._drip_timer = QTimer(self)
        self._drip_timer.timeout.connect(self._drip_tick)

        # mp3 兜底路径的状态：整句攒完写临时文件，交给 QMediaPlayer 顺序播放。
        self._mp3_buffer = bytearray()
        self._temp_dir = Path(tempfile.mkdtemp(prefix="miku-tts-"))
        self._temp_seq = 0
        self._play_queue: list[Path] = []
        self._current: Path | None = None
        self._playing = False

        self._audio_output = QAudioOutput(self)
        self._player = QMediaPlayer(self)
        self._player.setAudioOutput(self._audio_output)
        self._player.mediaStatusChanged.connect(self._on_media_status)
        self._player.errorOccurred.connect(self._on_player_error)

        self._configure_for_provider(provider)
        self._worker = _SynthWorker(provider, self)
        self._worker.audio_chunk_ready.connect(self._on_audio_chunk_ready)
        self._worker.sentence_done.connect(self._on_sentence_done)
        self._worker.start()

    def feed(self, text: str) -> None:
        """喂入一段流式增量文本（对应一次 ``ContentDelta``）。"""
        for sentence in self._buffer.feed(text):
            self._submit(sentence)

    def flush(self) -> None:
        """一轮回复结束（``LoopFinished``）时调用，把残留的最后一句补交合成。"""
        remainder = self._buffer.flush()
        if remainder is not None:
            self._submit(remainder)

    def _submit(self, sentence: str) -> None:
        """过滤掉 emoji/符号等念出来是噪音的字符后提交合成；清理后为空则跳过。"""
        speech = sanitize_for_speech(sentence)
        if speech:
            seq = self._next_seq
            self._next_seq += 1
            self._pending_sentences += 1
            self._worker.submit(self._generation, seq, speech)

    def stop(self) -> None:
        """打断当前播报：丢弃缓冲、在途音频与播放队列，立即静音。"""
        self._generation += 1
        self._buffer.flush()
        self._pending_sentences = 0

        self._drip_timer.stop()
        if self._sink is not None:
            self._sink.stop()
        self._sink_device = None
        self._sink_started = False
        self._backlog.clear()

        self._mp3_buffer.clear()
        self._player.stop()
        self._playing = False
        for path in self._play_queue:
            self._unlink(path)
        self._play_queue.clear()
        if self._current is not None:
            self._unlink(self._current)
            self._current = None

    def close(self) -> None:
        """退出时调用：停止 worker 线程与播放，清理临时文件。"""
        self.stop()
        self._worker.stop()
        self._worker.wait(3000)
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def set_provider(self, provider: TTSProvider) -> None:
        """运行时更换合成 provider（如切换角色绑定的声音）：打断当前播报，换一个新 worker。"""
        self.stop()
        self._worker.stop()
        self._worker.wait(3000)
        self._configure_for_provider(provider)
        self._worker = _SynthWorker(provider, self)
        self._worker.audio_chunk_ready.connect(self._on_audio_chunk_ready)
        self._worker.sentence_done.connect(self._on_sentence_done)
        self._worker.start()

    def _configure_for_provider(self, provider: TTSProvider) -> None:
        """按新 provider 的 ``pcm_format`` 更新采样参数；格式变了就重建 sink（不同
        ``QAudioFormat`` 不能复用同一个 ``QAudioSink`` 实例）。
        """
        new_format = provider.pcm_format
        if new_format != self._pcm_format and self._sink is not None:
            self._sink.stop()
            self._sink = None
            self._sink_device = None
        self._pcm_format = new_format
        if new_format is not None:
            self._prebuffer_threshold_bytes = int(
                new_format.sample_rate * new_format.channels * _BYTES_PER_SAMPLE
                * _PREBUFFER_SECONDS
            )
        else:
            self._prebuffer_threshold_bytes = 0

    # ---- PCM 播放路径 ----

    def _ensure_sink(self) -> QAudioSink:
        if self._sink is None:
            pcm_format = self._pcm_format
            assert pcm_format is not None
            audio_format = QAudioFormat()
            audio_format.setSampleRate(pcm_format.sample_rate)
            audio_format.setChannelCount(pcm_format.channels)
            audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            self._sink = QAudioSink(audio_format, self)
        return self._sink

    def _start_pcm_playback(self) -> None:
        sink = self._ensure_sink()
        device = sink.start()
        if device is None:
            logger.warning("QAudioSink.start() 未返回可写设备，放弃本轮播放")
            return
        self._sink_started = True
        self._sink_device = device
        self._drip_timer.start(_DRIP_INTERVAL_MS)
        self._drip_tick()

    def _drip_tick(self) -> None:
        device = self._sink_device
        sink = self._sink
        if device is None or sink is None:
            self._drip_timer.stop()
            return
        free = sink.bytesFree()
        if free > 0 and self._backlog:
            n = min(free, len(self._backlog))
            device.write(bytes(self._backlog[:n]))
            del self._backlog[:n]
        if not self._backlog and self._pending_sentences == 0:
            self._drip_timer.stop()

    # ---- 信号回调 ----

    def _on_audio_chunk_ready(self, generation: int, seq: int, chunk: bytes) -> None:
        if generation != self._generation:
            # 属于已被 stop() 打断的上一轮回复，丢弃。
            return
        if self._pcm_format is not None:
            self._backlog.extend(chunk)
            if not self._sink_started:
                if len(self._backlog) >= self._prebuffer_threshold_bytes:
                    self._start_pcm_playback()
            elif not self._drip_timer.isActive():
                self._drip_timer.start(_DRIP_INTERVAL_MS)
        else:
            self._mp3_buffer.extend(chunk)

    def _on_sentence_done(self, generation: int, seq: int) -> None:
        if generation != self._generation:
            return
        self._pending_sentences = max(0, self._pending_sentences - 1)
        if self._pcm_format is not None:
            if not self._sink_started and self._backlog:
                # 整句音频比冷启动预缓冲阈值还短：不再等更多字节，立刻出声。
                self._start_pcm_playback()
        elif self._mp3_buffer:
            self._flush_mp3_buffer()

    # ---- mp3 兜底播放路径 ----

    def _flush_mp3_buffer(self) -> None:
        path = self._temp_dir / f"{self._temp_seq}.mp3"
        self._temp_seq += 1
        try:
            path.write_bytes(self._mp3_buffer)
        except OSError:
            logger.exception("写入 TTS 临时音频失败")
            self._mp3_buffer.clear()
            return
        self._mp3_buffer.clear()
        self._play_queue.append(path)
        self._pump()

    def _pump(self) -> None:
        if self._playing or not self._play_queue:
            return
        self._current = self._play_queue.pop(0)
        self._playing = True
        self._player.setSource(QUrl.fromLocalFile(str(self._current)))
        self._player.play()

    def _on_media_status(self, status: QMediaPlayer.MediaStatus) -> None:
        if status in (
            QMediaPlayer.MediaStatus.EndOfMedia,
            QMediaPlayer.MediaStatus.InvalidMedia,
        ):
            self._advance()

    def _on_player_error(self, error: QMediaPlayer.Error, error_string: str) -> None:
        if error != QMediaPlayer.Error.NoError:
            logger.warning("TTS 播放出错：%s", error_string)
            self._advance()

    def _advance(self) -> None:
        """当前句子播放结束/失败：清理它并推进到下一句。"""
        if not self._playing:
            return
        self._playing = False
        if self._current is not None:
            self._unlink(self._current)
            self._current = None
        self._pump()

    @staticmethod
    def _unlink(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.debug("清理 TTS 临时文件失败：%s", path, exc_info=True)
