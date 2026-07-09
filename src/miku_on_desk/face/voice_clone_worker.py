"""在后台线程跑一次 ElevenLabs IVC 声音克隆调用，把结果通过 Qt 信号回传主线程。

跟 ``CharacterGenerationWorker`` 同款约定：``QThread`` 子类，一次性任务，取消用
``threading.Event``。IVC 是单次秒级 HTTP 调用，拿到结果后才有机会检查取消标记
（无法在调用中途真正打断网络请求）。
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, QThread, Signal

from miku_on_desk.brain.tts.voice_clone import VoiceCloneConfig, VoiceCloneError, clone_voice


class VoiceCloneWorker(QThread):
    """一次性任务：构造后调用 ``start()``，通过信号获知进展与终态，不可复用。"""

    finished_ok = Signal(str)  # voice_id
    failed = Signal(str)

    def __init__(self, config: VoiceCloneConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._cancel_requested = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def run(self) -> None:
        try:
            voice_id = clone_voice(self._config)
        except VoiceCloneError as exc:
            if not self._cancel_requested.is_set():
                self.failed.emit(str(exc))
            return
        if not self._cancel_requested.is_set():
            self.finished_ok.emit(voice_id)
