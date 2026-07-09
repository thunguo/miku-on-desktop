"""ElevenLabs 实时流式 STT 实现：走官方 ``elevenlabs`` SDK 的
``speech_to_text.realtime.connect`` 建立 WebSocket 会话，逐块喂入 PCM，通过事件回调拿到
增量转写结果。

两个必须遵守的 SDK 约束（已直接阅读已安装 SDK 源码 ``elevenlabs/realtime/connection.py``
确认）：

1. ``RealtimeConnection`` 的事件分发是**同步**调用注册的回调；回调内抛出的异常只会被 SDK
   内部 ``print()`` 吞掉，不会传播、也不会走本项目的 logger。因此这里注册给 SDK 的每个
   lambda 都包一层 ``try/except``，自行记录日志后再委托给 ``STTProvider`` 契约的回调。
2. 各类具体错误事件（``AUTH_ERROR``/``QUOTA_EXCEEDED``/``RATE_LIMITED`` 等）在 SDK 内部
   会连带触发一次通用的 ``RealtimeEvents.ERROR``，因此只订阅 ``ERROR`` 这一个事件就能覆盖
   所有错误子类型。

``PARTIAL_TRANSCRIPT``/``COMMITTED_TRANSCRIPT`` 的回调参数是普通 dict（不是 SDK 里的
Pydantic payload 类型，那些类型运行时从未被实际构造），文本字段是 ``"text"``；``ERROR``
回调参数的错误信息字段是 ``"error"``。``CLOSE`` 事件不带任何参数。
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable
from typing import Any

from elevenlabs import AudioFormat, CommitStrategy, RealtimeEvents
from elevenlabs.client import AsyncElevenLabs
from elevenlabs.realtime.connection import RealtimeConnection
from elevenlabs.realtime.scribe import RealtimeAudioOptions

from miku_on_desk.config.settings import VoiceInputConfig

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000


class ElevenLabsSTTSession:
    """包装一个已建立的 :class:`RealtimeConnection`，暴露 :class:`STTSession` 契约。"""

    def __init__(self, connection: RealtimeConnection) -> None:
        self._connection = connection

    async def send_chunk(self, pcm: bytes) -> None:
        await self._connection.send({"audio_base_64": base64.b64encode(pcm).decode("ascii")})

    async def close(self) -> None:
        await self._connection.close()


class ElevenLabsSTTProvider:
    """把实时音频流转写为文本的 :class:`~miku_on_desk.brain.stt.base.STTProvider` 实现。"""

    def __init__(self, config: VoiceInputConfig) -> None:
        if not config.api_key:
            raise ValueError("ElevenLabs 语音输入需要配置 api_key")
        self._config = config
        self._client = AsyncElevenLabs(api_key=config.api_key, base_url=config.base_url or None)

    async def open_session(
        self,
        *,
        on_partial: Callable[[str], None],
        on_committed: Callable[[str], None],
        on_error: Callable[[str], None],
        on_close: Callable[[], None],
    ) -> ElevenLabsSTTSession:
        options: RealtimeAudioOptions = {
            "model_id": self._config.model_id,
            "audio_format": AudioFormat.PCM_16000,
            "sample_rate": _SAMPLE_RATE,
            "commit_strategy": CommitStrategy.VAD,
        }
        if self._config.language_code:
            options["language_code"] = self._config.language_code

        connection = await self._client.speech_to_text.realtime.connect(options)

        def _handle_partial(data: dict[str, Any]) -> None:
            try:
                on_partial(data.get("text", ""))
            except Exception:
                logger.exception("处理 partial_transcript 回调失败")

        def _handle_committed(data: dict[str, Any]) -> None:
            try:
                on_committed(data.get("text", ""))
            except Exception:
                logger.exception("处理 committed_transcript 回调失败")

        def _handle_error(data: dict[str, Any]) -> None:
            try:
                on_error(data.get("error", "未知错误"))
            except Exception:
                logger.exception("处理语音输入 error 回调失败")

        def _handle_close() -> None:
            try:
                on_close()
            except Exception:
                logger.exception("处理语音输入 close 回调失败")

        connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, _handle_partial)
        connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, _handle_committed)
        connection.on(RealtimeEvents.ERROR, _handle_error)
        connection.on(RealtimeEvents.CLOSE, _handle_close)

        return ElevenLabsSTTSession(connection)
