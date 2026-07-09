"""在后台线程生成一段朗读文本，把结果通过 Qt 信号回传主线程。

跟 ``CharacterGenerationWorker``/``VoiceCloneWorker`` 同款约定：``QThread`` 子类，一次性任务。
``build_providers``/``ModelRouter`` 在线程内部构造——不能复用 ``_brain_main`` 线程里的那份实例
（线程不安全），也不能 import ``main.py``（会与其已经导入的 UI 组件循环引用）。
"""

from __future__ import annotations

import asyncio
import threading

from PySide6.QtCore import QObject, QThread, Signal

from miku_on_desk.brain.model_router import ModelRouter
from miku_on_desk.brain.provider_factory import build_providers
from miku_on_desk.brain.reading_script import generate_reading_script
from miku_on_desk.config.settings import ModelRouterConfig


class ReadingScriptWorker(QThread):
    """一次性任务：构造后调用 ``start()``，通过信号获知进展与终态，不可复用。"""

    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        description: str,
        model_router_config: ModelRouterConfig,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._description = description
        self._model_router_config = model_router_config
        self._cancel_requested = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def run(self) -> None:
        try:
            providers = build_providers(self._model_router_config)
            router = ModelRouter(self._model_router_config)
            text = asyncio.run(
                generate_reading_script(
                    description=self._description, router=router, providers=providers
                )
            )
        except Exception as exc:
            if not self._cancel_requested.is_set():
                self.failed.emit(str(exc))
            return
        if not self._cancel_requested.is_set():
            self.finished_ok.emit(text)
