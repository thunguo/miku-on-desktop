"""语音转文字（STT）：把用户说话的音频实时流式转写为文本。

分层与 ``brain/tts`` 对称：转写本身是"无 UI 依赖的纯 asyncio 逻辑"，放在 ``brain`` 侧
（``STTProvider``/``STTSession`` 抽象 + ``ElevenLabsSTTProvider`` 实现）；音频采集与
跨线程编排涉及 Qt 多媒体与线程，属于 UI 关注点，放在 ``face/ui/audio_capture.py`` 与
``face/stt_worker.py``。这样换 STT 实现只需新增一个 ``STTProvider``，不触碰采集/线程链路。
"""
