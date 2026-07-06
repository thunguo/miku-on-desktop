"""``CharacterGenerationWorker`` 的信号路由回归测试。

直接同步调用 ``.run()``（而非 ``.start()``）避免真实多线程的不确定性——测试关心的是
"``generate_character`` 的三种结局分别路由到哪个信号"，不需要真的起一个 OS 线程。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from miku_on_desk.character_generation import GenerationCancelled, GenerationConfig
from miku_on_desk.face import character_generation_worker as worker_module
from miku_on_desk.face.character_generation_worker import CharacterGenerationWorker
from miku_on_desk.face.pet_state import PetState
from miku_on_desk.face.sprite_sheet import SpriteSheetMeta, StateSpriteInfo


def _config(tmp_path: Path) -> GenerationConfig:
    return GenerationConfig(
        pet_name="test_pet",
        description="a test character",
        output_dir=tmp_path / "test_pet",
        api_key="sk-test",
    )


def _meta() -> SpriteSheetMeta:
    return SpriteSheetMeta(
        pet_name="test_pet",
        frame_width=4,
        frame_height=4,
        columns=1,
        rows=1,
        fallback_state=PetState.IDLE,
        states={PetState.IDLE: StateSpriteInfo(row=0, frame_count=1, fps=1.0, loop=True)},
    )


def _patch_generate_character(
    monkeypatch: pytest.MonkeyPatch, fn: Callable[..., Any]
) -> None:
    monkeypatch.setattr(worker_module, "generate_character", fn)


def test_run_emits_finished_ok_on_success(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sheet = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    meta = _meta()

    def fake_generate_character(
        config: GenerationConfig, *, on_progress: Callable[[Any], None], should_cancel: Any
    ) -> tuple[Image.Image, SpriteSheetMeta, list[str]]:
        del config, should_cancel
        on_progress("progress-event")
        return sheet, meta, ["QA 提示"]

    _patch_generate_character(monkeypatch, fake_generate_character)

    progress_events: list[Any] = []
    finished_calls: list[tuple[Any, Any, Any]] = []
    failed_calls: list[str] = []
    cancelled_calls: list[None] = []

    worker = CharacterGenerationWorker(_config(tmp_path))
    worker.progress.connect(progress_events.append)
    worker.finished_ok.connect(lambda s, m, p: finished_calls.append((s, m, p)))
    worker.failed.connect(failed_calls.append)
    worker.cancelled.connect(lambda: cancelled_calls.append(None))

    worker.run()

    assert progress_events == ["progress-event"]
    assert len(finished_calls) == 1
    assert finished_calls[0][0] is sheet
    assert finished_calls[0][1] is meta
    assert finished_calls[0][2] == ["QA 提示"]
    assert failed_calls == []
    assert cancelled_calls == []


def test_run_emits_cancelled_on_generation_cancelled(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_generate_character(
        config: GenerationConfig, *, on_progress: Callable[[Any], None], should_cancel: Any
    ) -> tuple[Image.Image, SpriteSheetMeta, list[str]]:
        del config, on_progress, should_cancel
        raise GenerationCancelled()

    _patch_generate_character(monkeypatch, fake_generate_character)

    finished_calls: list[None] = []
    failed_calls: list[str] = []
    cancelled_calls: list[None] = []

    worker = CharacterGenerationWorker(_config(tmp_path))
    worker.finished_ok.connect(lambda *_: finished_calls.append(None))
    worker.failed.connect(failed_calls.append)
    worker.cancelled.connect(lambda: cancelled_calls.append(None))

    worker.run()

    assert cancelled_calls == [None]
    assert finished_calls == []
    assert failed_calls == []


def test_run_emits_failed_on_unexpected_exception(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_generate_character(
        config: GenerationConfig, *, on_progress: Callable[[Any], None], should_cancel: Any
    ) -> tuple[Image.Image, SpriteSheetMeta, list[str]]:
        del config, on_progress, should_cancel
        raise RuntimeError("API 超时")

    _patch_generate_character(monkeypatch, fake_generate_character)

    finished_calls: list[None] = []
    failed_calls: list[str] = []
    cancelled_calls: list[None] = []

    worker = CharacterGenerationWorker(_config(tmp_path))
    worker.finished_ok.connect(lambda *_: finished_calls.append(None))
    worker.failed.connect(failed_calls.append)
    worker.cancelled.connect(lambda: cancelled_calls.append(None))

    worker.run()

    assert failed_calls == ["API 超时"]
    assert finished_calls == []
    assert cancelled_calls == []


def test_request_cancel_sets_should_cancel_flag_observed_by_generate_character(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_should_cancel: list[bool] = []

    def fake_generate_character(
        config: GenerationConfig, *, on_progress: Callable[[Any], None], should_cancel: Any
    ) -> tuple[Image.Image, SpriteSheetMeta, list[str]]:
        del config, on_progress
        observed_should_cancel.append(should_cancel())
        raise GenerationCancelled()

    _patch_generate_character(monkeypatch, fake_generate_character)

    worker = CharacterGenerationWorker(_config(tmp_path))
    worker.request_cancel()
    worker.run()

    assert observed_should_cancel == [True]
