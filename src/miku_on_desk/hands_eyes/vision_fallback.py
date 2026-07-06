"""``screen_analyze`` 网格标注兜底的图像编码工具：只暴露裸的 ``(media_type, base64_data)``
数据，不引入 `brain/` 的任何类型（如 `ImageBlock`）——把这份数据包装成 Provider 能接受的
消息、发起实际的视觉 LLM 调用，是 `brain/tools/builtin/screen_analyze.py` 的职责。这样
`hands_eyes/` 完全不需要知道"LLM"这个概念，保持这一层对 Brain 的单向依赖（Brain 依赖
hands_eyes，反之不成立）。
"""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

_IMAGE_FORMAT = "PNG"
MEDIA_TYPE = "image/png"


def encode_image_as_base64(image: Image.Image) -> tuple[str, str]:
    """返回 ``(media_type, base64_data)``，media_type 恒为 :data:`MEDIA_TYPE`。

    供网格标注两轮编排（`brain/tools/builtin/screen_analyze.py`）复用——不止原始截图，
    叠加了网格线/裁剪放大后的中间图像也要走同一套编码逻辑。
    """
    buffer = BytesIO()
    image.save(buffer, format=_IMAGE_FORMAT)
    return MEDIA_TYPE, base64.b64encode(buffer.getvalue()).decode("ascii")