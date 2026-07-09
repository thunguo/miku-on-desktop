"""STT Provider 抽象：把音频 PCM 流实时转写为文本，通过回调交付结果。

用 ``Protocol`` 而非 ABC——与 ``brain/tts/base.py::TTSProvider`` 同样的理由：调用方
（``face/stt_worker.py::SttWorker``）只依赖"能开一个会话、能喂 PCM、能收到转写结果"这一
结构，不需要继承关系，也便于测试传入任意假实现。

``on_error``/``on_close`` 拆成两个独立回调而不是共用一个"结束"回调——区分"服务端报错"
（连接可能仍活着，或已被服务端关闭并附带原因）与"连接正常/被动关闭"（没有错误信息），上层
要用这个区别决定是否展示错误文案。四个回调全部是**同步函数**：底层 ElevenLabs SDK 的事件
分发是同步调用，异步 provider 实现内部会在同步上下文里调用它们，不能假设回调本身是协程。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class STTSession(Protocol):
    async def send_chunk(self, pcm: bytes) -> None:
        """把一段裸 PCM 音频数据发送给转写服务；``pcm`` 保证非空。"""
        ...

    async def close(self) -> None:
        """主动结束本次转写会话；会话结束后不可再调用 ``send_chunk``。"""
        ...


@runtime_checkable
class STTProvider(Protocol):
    async def open_session(
        self,
        *,
        on_partial: Callable[[str], None],
        on_committed: Callable[[str], None],
        on_error: Callable[[str], None],
        on_close: Callable[[], None],
    ) -> STTSession:
        """开启一次实时转写会话，注册四个同步回调，返回可持续喂音频的 :class:`STTSession`。

        ``on_partial``——收到未最终确定的中间转写结果（同一句话后续可能被覆盖）。
        ``on_committed``——收到服务端已确认最终的转写文本片段。
        ``on_error``——转写服务报告错误（鉴权/限流/配额等），会话可能仍然存活。
        ``on_close``——连接已关闭，无论是本方主动关闭还是对端/网络原因。
        """
        ...
