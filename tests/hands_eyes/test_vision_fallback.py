"""vision_fallback 的回归测试：图像编码是纯函数，不涉及截图/OS 调用。"""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from miku_on_desk.hands_eyes.vision_fallback import MEDIA_TYPE, encode_image_as_base64


def test_encode_image_as_base64_returns_media_type_and_encoded_png() -> None:
    image = Image.new("RGB", (3, 2), color=(1, 2, 3))

    media_type, data = encode_image_as_base64(image)

    assert media_type == MEDIA_TYPE == "image/png"
    decoded = Image.open(BytesIO(base64.b64decode(data)))
    assert decoded.size == (3, 2)
