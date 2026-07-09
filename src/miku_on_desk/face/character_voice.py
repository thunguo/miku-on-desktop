"""按角色（`pet_dir`）绑定专属声音的 sidecar 文件（`voice.json`）读写与解析。

不塞进 `pet.json`——`SpriteSheetMeta` 的语义是"精灵表描述"，校验严格，混入声音字段会污染它；
独立文件让"没绑定声音"直接等于"文件不存在"，老角色零行为变化。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ValidationError

from miku_on_desk.config.settings import AppSettings, TTSConfig, TTSProviderName

_VOICE_FILENAME = "voice.json"


class PetVoiceConfigError(Exception):
    """`voice.json` 内容未通过解析。"""


class PetVoiceConfig(BaseModel):
    provider: TTSProviderName
    voice: str
    model: str = "tts-1"
    rate: str = "+0%"
    volume: str = "+0%"


def _voice_path(pet_dir: Path) -> Path:
    return pet_dir / _VOICE_FILENAME


def load_pet_voice_config(pet_dir: Path) -> PetVoiceConfig | None:
    """文件不存在返回 None（表示"沿用全局声音"）；存在但解析失败则抛错，不静默吞掉。"""
    path = _voice_path(pet_dir)
    if not path.exists():
        return None
    try:
        return PetVoiceConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise PetVoiceConfigError(f"解析 {path} 失败：{exc}") from exc


def save_pet_voice_config(pet_dir: Path, config: PetVoiceConfig) -> None:
    _voice_path(pet_dir).write_text(config.model_dump_json(indent=2), encoding="utf-8")


def delete_pet_voice_config(pet_dir: Path) -> None:
    """"恢复默认声音"用：文件不存在时静默跳过。"""
    _voice_path(pet_dir).unlink(missing_ok=True)


def resolve_tts_config_for_pet(pet_dir: Path, settings: AppSettings) -> TTSConfig:
    """给定角色目录，算出实际应使用的 `TTSConfig`。

    没有 `voice.json` 时原样返回 `settings.tts`。绑定了专属声音时，`enabled` 恒等于
    `settings.tts.enabled`——每角色声音只决定"用哪个声音"，不决定"要不要说话"，全局开关
    仍是唯一总闸。密钥按 `provider` 决定来源：ElevenLabs 走 `settings.voice_cloning`
    （跟全局 TTS provider 选型无关的独立凭证位），OpenAI 走 `settings.tts`，Edge 无需密钥。
    """
    voice_config = load_pet_voice_config(pet_dir)
    if voice_config is None:
        return settings.tts

    api_key: str | None
    base_url: str | None
    if voice_config.provider is TTSProviderName.ELEVENLABS:
        api_key = settings.voice_cloning.elevenlabs_api_key
        base_url = settings.voice_cloning.elevenlabs_base_url
    elif voice_config.provider is TTSProviderName.OPENAI:
        api_key = settings.tts.api_key
        base_url = settings.tts.base_url
    else:
        api_key = None
        base_url = None

    return TTSConfig(
        enabled=settings.tts.enabled,
        provider=voice_config.provider,
        voice=voice_config.voice,
        rate=voice_config.rate,
        volume=voice_config.volume,
        api_key=api_key,
        base_url=base_url,
        model=voice_config.model,
    )
