"""把 Miku 的流式回复文本变成顺序播放的语音。

三段职责，都汇聚在 UI（Qt 主）线程上的 :class:`SpeechController`：

1. **断句**——``feed`` 把 ``ContentDelta`` 增量喂给 :class:`SentenceBuffer`，凑成整句才提交，
   避免逐字合成造成的破碎语音；``flush`` 在一轮回复结束时补上最后没有标点收尾的残句。
2. **合成**——句子提交给后台 :class:`_SynthWorker` 线程（TTS 是网络 IO，不能阻塞 UI 线程）。
   worker 单线程 FIFO 逐句合成，因此音频天然按提交顺序返回，播放顺序无需额外排序。
3. **播放**——合成好的音频写成临时文件，用 ``QMediaPlayer`` 串行播放（一句放完再放下一句），
   ``QMediaPlayer`` 必须在创建它的线程使用，所以播放留在 UI 线程。

``stop`` 用一个"代际"计数器丢弃所有在途/排队的音频：打断时递增代际，之前提交的句子即便合成
完成、其携带的旧代际也会在回调里被识别为过期而丢弃，不会串到下一轮回复里。
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from miku_on_desk.brain.tts.base import TTSProvider
from miku_on_desk.brain.tts.sentence_buffer import SentenceBuffer
from miku_on_desk.brain.tts.text_sanitizer import sanitize_for_speech

logger = logging.getLogger(__name__)


class _SynthWorker(QThread):
    """后台合成线程：从队列取句子，调用 provider 合成，把音频字节发回 UI 线程。

    ``audio_ready`` 从 ``run()`` 所在的 worker 线程发出，经 Qt 默认的 AutoConnection 排队投递
    到 UI 线程的槽——这正是让网络合成不阻塞 UI、又不必手写线程同步的关键。
    """

    audio_ready = Signal(int, bytes)  # (generation, audio_bytes)

    _STOP = (None, None, None)

    def __init__(self, provider: TTSProvider, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._provider = provider
        # 简单的线程安全 FIFO；用标准库 queue 即可，不需要 asyncio 机制。
        import queue

        self._queue: queue.Queue[tuple[int | None, str | None, object]] = queue.Queue()

    def submit(self, generation: int, text: str) -> None:
        self._queue.put((generation, text, None))

    def stop(self) -> None:
        self._queue.put(self._STOP)

    def run(self) -> None:
        import asyncio

        while True:
            generation, text, _ = self._queue.get()
            if text is None or generation is None:
                return
            try:
                audio = asyncio.run(self._provider.synthesize(text))
            except Exception:
                logger.exception("TTS 合成失败，跳过该句：%r", text)
                continue
            if audio:
                self.audio_ready.emit(generation, audio)


class SpeechController(QObject):
    """UI 线程上的语音控制器：喂文本进来，它负责断句、后台合成、顺序播放。"""

    def __init__(self, provider: TTSProvider, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._buffer = SentenceBuffer()
        self._generation = 0

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

        self._worker = _SynthWorker(provider, self)
        self._worker.audio_ready.connect(self._on_audio_ready)
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
            self._worker.submit(self._generation, speech)

    def stop(self) -> None:
        """打断当前播报：丢弃缓冲、在途音频与播放队列，立即静音。"""
        self._generation += 1
        self._buffer.flush()
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
        import shutil

        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def _on_audio_ready(self, generation: int, audio: bytes) -> None:
        if generation != self._generation:
            # 属于已被 stop() 打断的上一轮回复，丢弃。
            return
        path = self._temp_dir / f"{self._temp_seq}.mp3"
        self._temp_seq += 1
        try:
            path.write_bytes(audio)
        except OSError:
            logger.exception("写入 TTS 临时音频失败")
            return
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
