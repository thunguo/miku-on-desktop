"""跨平台屏幕截图：mss 本身已经处理了 Windows/macOS 的差异，不需要再分平台实现。

`capture_screen()` 始终截取全部显示器拼接成的一张图（而非仅主显示器），`capture_origin()`
给出该图像素 (0, 0) 对应的全局桌面坐标——AX/UIA 返回的窗口坐标、`pynput` 点击坐标都是这个
全局坐标系，`crop_to_bounds()` 依赖这个 origin 把两者对齐，让 ROI 裁剪在多显示器环境下也
能正确工作（副屏应用不会因为"只截了主屏"而截不到）。
"""

from __future__ import annotations

import mss
from PIL import Image


def capture_screen() -> Image.Image:
    """截取所有显示器拼接成的一张图；像素 (0, 0) 对应的全局坐标见 :func:`capture_origin`。"""
    with mss.MSS() as sct:
        shot = sct.grab(sct.monitors[0])
        return Image.frombytes("RGB", shot.size, shot.rgb)


def capture_origin() -> tuple[int, int]:
    """返回 `capture_screen()` 图像像素 (0, 0) 对应的全局桌面坐标（多屏时可能非零/为负）。"""
    with mss.MSS() as sct:
        monitor = sct.monitors[0]
        return monitor["left"], monitor["top"]


def crop_to_bounds(
    image: Image.Image, bounds: tuple[int, int, int, int], origin: tuple[int, int]
) -> tuple[Image.Image, tuple[int, int]]:
    """按窗口边界裁剪。

    ``bounds`` 是全局桌面坐标下的 (x, y, width, height)（AX/UIA 直接返回的格式，调用方不
    需要预先做任何换算）；``origin`` 是 ``image`` 对应的 :func:`capture_origin` 结果。

    返回 (裁剪后的图像, 偏移量)，偏移量**始终是全局桌面坐标**——不管有没有实际发生裁剪：
    未裁剪（或越界/零面积退化）时就是 ``origin`` 本身，裁剪成功时是 ``origin`` 叠加裁剪区域
    左上角在原图中的像素位置。调用方只需要把这个偏移量直接加到裁剪图内算出的任何相对坐标
    上，就能得到可以直接喂给 ``pynput`` 点击的全局坐标，不需要关心中间的局部/全局坐标系
    换算细节。
    """
    ox, oy = origin
    x, y, w, h = bounds
    local_x, local_y = x - ox, y - oy
    left = max(0, min(local_x, image.width))
    top = max(0, min(local_y, image.height))
    right = max(left, min(local_x + w, image.width))
    bottom = max(top, min(local_y + h, image.height))
    if right <= left or bottom <= top:
        return image, (ox, oy)
    return image.crop((left, top, right, bottom)), (ox + left, oy + top)
