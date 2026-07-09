"""``clone_voice`` 的参数拼装、返回值解析与异常包装回归测试。

用假 ``ElevenLabs`` client 替换真实 SDK（monkeypatch 模块级引用），不发真实网络请求。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from miku_on_desk.brain.tts import voice_clone as voice_clone_module
from miku_on_desk.brain.tts.voice_clone import VoiceCloneConfig, VoiceCloneError, clone_voice


@dataclass
class _FakeIvcResponse:
    voice_id: str


class _FakeIvcClient:
    def __init__(self, response: _FakeIvcResponse | Exception) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeIvcResponse:
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeVoices:
    def __init__(self, ivc: _FakeIvcClient) -> None:
        self.ivc = ivc


class _FakeElevenLabs:
    last_instance: _FakeElevenLabs | None = None

    def __init__(self, *, api_key: str, base_url: str | None) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.voices = _FakeVoices(_FakeIvcClient(_FakeIvcResponse(voice_id="voice-abc")))
        _FakeElevenLabs.last_instance = self


def _config(**overrides: Any) -> VoiceCloneConfig:
    defaults: dict[str, Any] = dict(
        name="测试角色",
        audio_bytes=b"RIFF....WAVEfmt ",
        api_key="sk-elevenlabs",
    )
    defaults.update(overrides)
    return VoiceCloneConfig(**defaults)


def test_clone_voice_returns_voice_id_from_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voice_clone_module, "ElevenLabs", _FakeElevenLabs)

    voice_id = clone_voice(_config())

    assert voice_id == "voice-abc"


def test_clone_voice_passes_name_files_and_options_to_ivc_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice_clone_module, "ElevenLabs", _FakeElevenLabs)

    clone_voice(
        _config(
            audio_filename="clip.wav",
            remove_background_noise=False,
            description="一个测试声音",
        )
    )

    instance = _FakeElevenLabs.last_instance
    assert instance is not None
    call = instance.voices.ivc.calls[0]
    assert call["name"] == "测试角色"
    assert call["files"] == [("clip.wav", b"RIFF....WAVEfmt ", "audio/wav")]
    assert call["remove_background_noise"] is False
    assert call["description"] == "一个测试声音"


def test_clone_voice_uses_api_key_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voice_clone_module, "ElevenLabs", _FakeElevenLabs)

    clone_voice(_config(base_url="https://elevenlabs.example.com"))

    instance = _FakeElevenLabs.last_instance
    assert instance is not None
    assert instance.api_key == "sk-elevenlabs"
    assert instance.base_url == "https://elevenlabs.example.com"


def test_clone_voice_strips_trailing_v1_from_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voice_clone_module, "ElevenLabs", _FakeElevenLabs)

    clone_voice(_config(base_url="https://elevenlabs.example.com/v1"))

    instance = _FakeElevenLabs.last_instance
    assert instance is not None
    assert instance.base_url == "https://elevenlabs.example.com"


def test_clone_voice_strips_trailing_slash_and_v1_from_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voice_clone_module, "ElevenLabs", _FakeElevenLabs)

    clone_voice(_config(base_url="https://elevenlabs.example.com/v1/"))

    instance = _FakeElevenLabs.last_instance
    assert instance is not None
    assert instance.base_url == "https://elevenlabs.example.com"


def test_clone_voice_passes_none_base_url_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voice_clone_module, "ElevenLabs", _FakeElevenLabs)

    clone_voice(_config(base_url=None))

    instance = _FakeElevenLabs.last_instance
    assert instance is not None
    assert instance.base_url is None


def test_clone_voice_wraps_sdk_exception_into_voice_clone_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingElevenLabs(_FakeElevenLabs):
        def __init__(self, *, api_key: str, base_url: str | None) -> None:
            super().__init__(api_key=api_key, base_url=base_url)
            self.voices = _FakeVoices(_FakeIvcClient(RuntimeError("素材格式不支持")))

    monkeypatch.setattr(voice_clone_module, "ElevenLabs", _FailingElevenLabs)

    with pytest.raises(VoiceCloneError, match="素材格式不支持"):
        clone_voice(_config())
