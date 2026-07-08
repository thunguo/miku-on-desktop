"""文字转语音（TTS）：把 Miku 的回复文本合成为可播放的音频。

分层与项目其余部分一致：合成本身是"无 UI 依赖的纯 asyncio 逻辑"，放在 ``brain`` 侧
（``TTSProvider`` 抽象 + ``EdgeTTSProvider`` 实现 + 纯函数式的 ``SentenceBuffer`` 断句）；
真正的排队合成与顺序播放涉及 Qt 多媒体与线程，属于 UI 关注点，放在
``face/ui/speech_controller.py``。这样换 TTS 实现（如本地 GPT-SoVITS）只需新增一个
``TTSProvider``，不触碰播放链路。
"""
