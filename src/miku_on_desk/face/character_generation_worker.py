"""在后台线程跑完整的角色生成流水线，把进度/结果通过 Qt 信号编组回主线程。

用 ``QThread`` 子类而非裸 ``threading.Thread``：子类自带的 ``Signal`` 天然获得
``BrainEventBus`` 依赖的同款跨线程 ``QueuedConnection`` 编组，比额外包一层 ``QObject``
总线更省代码。取消用 ``threading.Event``：参考图这次阻塞调用之后、以及各状态动作条阶段
每提交/完成一个就检查一次——`generate_character` 内部把动作条阶段改成了并发（最多几路
同时请求），取消依然不打断已经在飞的调用，只是不再等它、不再排新的；每次调用耗时
10-30 秒，不要求硬实时。
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, QThread, Signal

from miku_on_desk.character_generation import (
    GenerationCancelled,
    GenerationConfig,
    generate_character,
)


class CharacterGenerationWorker(QThread):
    """一次性任务：构造后调用 ``start()``，通过信号获知进展与终态，不可复用。"""

    progress = Signal(object)  # GenerationProgress
    finished_ok = Signal(object, object, object)  # (Image.Image sheet, SpriteSheetMeta, problems)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, config: GenerationConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._cancel_requested = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def run(self) -> None:
        try:
            sheet, meta, problems = generate_character(
                self._config,
                on_progress=lambda p: self.progress.emit(p),
                should_cancel=self._cancel_requested.is_set,
            )
        except GenerationCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(sheet, meta, problems)
