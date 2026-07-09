"""voice.json 读写往返、异常路径，以及 resolve_tts_config_for_pet 的密钥来源正确性。"""

from __future__ import annotations

from pathlib import Path

import pytest

from miku_on_desk.config.settings import AppSettings, TTSConfig, TTSProviderName, VoiceCloningConfig
from miku_on_desk.face.character_voice import (
    PetVoiceConfig,
    PetVoiceConfigError,
    delete_pet_voice_config,
    load_pet_voice_config,
    resolve_tts_config_for_pet,
    save_pet_voice_config,
)


def test_load_pet_voice_config_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert load_pet_voice_config(tmp_path) is None


def test_save_then_load_pet_voice_config_roundtrips(tmp_path: Path) -> None:
    config = PetVoiceConfig(provider=TTSProviderName.ELEVENLABS, voice="voice-abc")

    save_pet_voice_config(tmp_path, config)
    loaded = load_pet_voice_config(tmp_path)

    assert loaded == config
    assert (tmp_path / "voice.json").exists()


def test_load_pet_voice_config_rejects_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "voice.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(PetVoiceConfigError):
        load_pet_voice_config(tmp_path)


def test_delete_pet_voice_config_removes_file(tmp_path: Path) -> None:
    save_pet_voice_config(
        tmp_path, PetVoiceConfig(provider=TTSProviderName.EDGE, voice="zh-CN-XiaoxiaoNeural")
    )

    delete_pet_voice_config(tmp_path)

    assert load_pet_voice_config(tmp_path) is None


def test_delete_pet_voice_config_is_noop_when_file_missing(tmp_path: Path) -> None:
    delete_pet_voice_config(tmp_path)  # 不应抛错


def test_resolve_tts_config_for_pet_returns_global_tts_when_no_voice_json(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.tts = TTSConfig(enabled=True, provider=TTSProviderName.OPENAI, api_key="sk-global")

    resolved = resolve_tts_config_for_pet(tmp_path, settings)

    assert resolved == settings.tts


def test_resolve_tts_config_for_pet_elevenlabs_uses_voice_cloning_key_not_tts_key(
    tmp_path: Path,
) -> None:
    settings = AppSettings()
    settings.tts = TTSConfig(enabled=True, provider=TTSProviderName.OPENAI, api_key="sk-tts-key")
    settings.voice_cloning = VoiceCloningConfig(
        elevenlabs_api_key="sk-elevenlabs-key", elevenlabs_base_url="https://elevenlabs.example.com"
    )
    save_pet_voice_config(
        tmp_path, PetVoiceConfig(provider=TTSProviderName.ELEVENLABS, voice="voice-abc")
    )

    resolved = resolve_tts_config_for_pet(tmp_path, settings)

    assert resolved.provider is TTSProviderName.ELEVENLABS
    assert resolved.voice == "voice-abc"
    assert resolved.api_key == "sk-elevenlabs-key"
    assert resolved.base_url == "https://elevenlabs.example.com"


def test_resolve_tts_config_for_pet_openai_uses_tts_key_not_voice_cloning_key(
    tmp_path: Path,
) -> None:
    settings = AppSettings()
    settings.tts = TTSConfig(enabled=True, provider=TTSProviderName.EDGE, api_key="sk-tts-key")
    settings.voice_cloning = VoiceCloningConfig(elevenlabs_api_key="sk-elevenlabs-key")
    save_pet_voice_config(tmp_path, PetVoiceConfig(provider=TTSProviderName.OPENAI, voice="alloy"))

    resolved = resolve_tts_config_for_pet(tmp_path, settings)

    assert resolved.provider is TTSProviderName.OPENAI
    assert resolved.api_key == "sk-tts-key"


def test_resolve_tts_config_for_pet_edge_needs_no_api_key(tmp_path: Path) -> None:
    settings = AppSettings()
    save_pet_voice_config(
        tmp_path, PetVoiceConfig(provider=TTSProviderName.EDGE, voice="zh-CN-YunxiNeural")
    )

    resolved = resolve_tts_config_for_pet(tmp_path, settings)

    assert resolved.api_key is None
    assert resolved.base_url is None


def test_resolve_tts_config_for_pet_enabled_always_follows_global_tts_enabled(
    tmp_path: Path,
) -> None:
    settings = AppSettings()
    settings.tts = TTSConfig(enabled=False)
    save_pet_voice_config(tmp_path, PetVoiceConfig(provider=TTSProviderName.EDGE, voice="v"))

    resolved = resolve_tts_config_for_pet(tmp_path, settings)

    assert resolved.enabled is False

    settings.tts.enabled = True
    resolved_enabled = resolve_tts_config_for_pet(tmp_path, settings)

    assert resolved_enabled.enabled is True
