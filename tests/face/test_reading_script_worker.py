"""``ReadingScriptWorker`` 的信号路由回归测试。

直接同步调用 ``.run()``（而非 ``.start()``）避免真实多线程的不确定性——测试关心的是
"``build_providers``/``generate_reading_script`` 的成功/失败/取消三种结局分别路由到哪个信号"，
不需要真的起一个 OS 线程，也不需要真的调 LLM。
"""

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from miku_on_desk.config.settings import ModelRouterConfig
from miku_on_desk.face import reading_script_worker as worker_module
from miku_on_desk.face.reading_script_worker import ReadingScriptWorker


def _patch_build_providers(monkeypatch: pytest.MonkeyPatch, fn: Any) -> None:
    monkeypatch.setattr(worker_module, "build_providers", fn)


def _patch_generate_reading_script(monkeypatch: pytest.MonkeyPatch, fn: Any) -> None:
    monkeypatch.setattr(worker_module, "generate_reading_script", fn)


def test_run_emits_finished_ok_on_success(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_build_providers(monkeypatch, lambda config: {})

    async def fake_generate_reading_script(**kwargs: Any) -> str:
        del kwargs
        return "你今天心情怎么样呀"

    _patch_generate_reading_script(monkeypatch, fake_generate_reading_script)

    finished_calls: list[str] = []
    failed_calls: list[str] = []

    worker = ReadingScriptWorker("一个爱笑的猫娘", ModelRouterConfig())
    worker.finished_ok.connect(finished_calls.append)
    worker.failed.connect(failed_calls.append)

    worker.run()

    assert finished_calls == ["你今天心情怎么样呀"]
    assert failed_calls == []


def test_run_emits_failed_on_unexpected_exception(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_build_providers(config: ModelRouterConfig) -> dict[Any, Any]:
        del config
        raise RuntimeError("没有配置任何 Provider")

    _patch_build_providers(monkeypatch, fake_build_providers)

    finished_calls: list[str] = []
    failed_calls: list[str] = []

    worker = ReadingScriptWorker("一个爱笑的猫娘", ModelRouterConfig())
    worker.finished_ok.connect(finished_calls.append)
    worker.failed.connect(failed_calls.append)

    worker.run()

    assert failed_calls == ["没有配置任何 Provider"]
    assert finished_calls == []


def test_run_after_cancel_suppresses_finished_ok_signal(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_build_providers(monkeypatch, lambda config: {})

    async def fake_generate_reading_script(**kwargs: Any) -> str:
        del kwargs
        return "你今天心情怎么样呀"

    _patch_generate_reading_script(monkeypatch, fake_generate_reading_script)

    finished_calls: list[str] = []
    failed_calls: list[str] = []

    worker = ReadingScriptWorker("一个爱笑的猫娘", ModelRouterConfig())
    worker.finished_ok.connect(finished_calls.append)
    worker.failed.connect(failed_calls.append)
    worker.request_cancel()

    worker.run()

    assert finished_calls == []
    assert failed_calls == []


def test_run_after_cancel_suppresses_failed_signal(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_build_providers(config: ModelRouterConfig) -> dict[Any, Any]:
        del config
        raise RuntimeError("没有配置任何 Provider")

    _patch_build_providers(monkeypatch, fake_build_providers)

    finished_calls: list[str] = []
    failed_calls: list[str] = []

    worker = ReadingScriptWorker("一个爱笑的猫娘", ModelRouterConfig())
    worker.finished_ok.connect(finished_calls.append)
    worker.failed.connect(failed_calls.append)
    worker.request_cancel()

    worker.run()

    assert finished_calls == []
    assert failed_calls == []
