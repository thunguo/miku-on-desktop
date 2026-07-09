"""``VoiceCloneWorker`` 的信号路由回归测试。

直接同步调用 ``.run()``（而非 ``.start()``）避免真实多线程的不确定性——测试关心的是
"``clone_voice`` 的成功/失败/取消三种结局分别路由到哪个信号"，不需要真的起一个 OS 线程，
也不需要真的调 ElevenLabs。
"""

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from miku_on_desk.brain.tts.voice_clone import VoiceCloneConfig, VoiceCloneError
from miku_on_desk.face import voice_clone_worker as worker_module
from miku_on_desk.face.voice_clone_worker import VoiceCloneWorker


def _config() -> VoiceCloneConfig:
    return VoiceCloneConfig(
        name="测试角色",
        audio_bytes=b"RIFF....WAVEfmt ",
        api_key="sk-elevenlabs",
    )


def _patch_clone_voice(monkeypatch: pytest.MonkeyPatch, fn: Any) -> None:
    monkeypatch.setattr(worker_module, "clone_voice", fn)


def test_run_emits_finished_ok_on_success(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clone_voice(monkeypatch, lambda config: "voice-abc")

    finished_calls: list[str] = []
    failed_calls: list[str] = []

    worker = VoiceCloneWorker(_config())
    worker.finished_ok.connect(finished_calls.append)
    worker.failed.connect(failed_calls.append)

    worker.run()

    assert finished_calls == ["voice-abc"]
    assert failed_calls == []


def test_run_emits_failed_on_voice_clone_error(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_clone_voice(config: VoiceCloneConfig) -> str:
        del config
        raise VoiceCloneError("素材被拒绝")

    _patch_clone_voice(monkeypatch, fake_clone_voice)

    finished_calls: list[str] = []
    failed_calls: list[str] = []

    worker = VoiceCloneWorker(_config())
    worker.finished_ok.connect(finished_calls.append)
    worker.failed.connect(failed_calls.append)

    worker.run()

    assert failed_calls == ["素材被拒绝"]
    assert finished_calls == []


def test_run_after_cancel_suppresses_finished_ok_signal(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clone_voice(monkeypatch, lambda config: "voice-abc")

    finished_calls: list[str] = []
    failed_calls: list[str] = []

    worker = VoiceCloneWorker(_config())
    worker.finished_ok.connect(finished_calls.append)
    worker.failed.connect(failed_calls.append)
    worker.request_cancel()

    worker.run()

    assert finished_calls == []
    assert failed_calls == []


def test_run_after_cancel_suppresses_failed_signal(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_clone_voice(config: VoiceCloneConfig) -> str:
        del config
        raise VoiceCloneError("素材被拒绝")

    _patch_clone_voice(monkeypatch, fake_clone_voice)

    finished_calls: list[str] = []
    failed_calls: list[str] = []

    worker = VoiceCloneWorker(_config())
    worker.finished_ok.connect(finished_calls.append)
    worker.failed.connect(failed_calls.append)
    worker.request_cancel()

    worker.run()

    assert finished_calls == []
    assert failed_calls == []
