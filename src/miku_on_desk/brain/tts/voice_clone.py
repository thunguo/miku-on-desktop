"""ElevenLabs Instant Voice Cloning（IVC）：拿一段录音克隆出一个专属 voice_id。

用 IVC（``voices.ivc.create``）而不是 Professional Voice Cloning（PVC）——PVC 是为几十
分钟高质量录音 + 异步训练设计的，跟"录 30 秒、马上要角色"的产品节奏不匹配。IVC 是同步
单次 HTTP 调用，秒级返回，用同步 ``ElevenLabs`` client（不是 ``AsyncElevenLabs``）：调用方
在专属 ``QThread`` 里跑，没有事件循环需要复用。
"""

from __future__ import annotations

from dataclasses import dataclass

from elevenlabs.client import ElevenLabs


class VoiceCloneError(Exception):
    """ElevenLabs IVC 调用失败（网络错误、鉴权失败、素材被拒绝等）。"""


@dataclass(frozen=True)
class VoiceCloneConfig:
    name: str
    audio_bytes: bytes
    api_key: str
    audio_filename: str = "sample.wav"
    base_url: str | None = None
    remove_background_noise: bool = True
    description: str | None = None


def clone_voice(config: VoiceCloneConfig) -> str:
    """调用 IVC 克隆声音，返回新声音的 ``voice_id``。"""
    client = ElevenLabs(api_key=config.api_key, base_url=config.base_url or None)
    try:
        response = client.voices.ivc.create(
            name=config.name,
            files=[(config.audio_filename, config.audio_bytes, "audio/wav")],
            remove_background_noise=config.remove_background_noise,
            description=config.description,
        )
    except Exception as exc:
        raise VoiceCloneError(str(exc)) from exc
    return str(response.voice_id)
