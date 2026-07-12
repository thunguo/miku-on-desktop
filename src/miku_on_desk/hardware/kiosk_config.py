"""树莓派 kiosk 硬件端专属设置，挂在 ``AppSettings.kiosk`` 字段下，复用现成的 JSON
持久化（``AppSettings.load``/``save``）。只在 ``kiosk_main.py`` 入口下使用，桌面版
``main.py`` 不读这个字段。

不设 ``enabled`` 之类的开关字段——kiosk 模式与桌面模式不是同一进程里靠配置切换的
两条分支，而是靠"启动的是哪个入口命令"（``miku-on-desk`` 还是 ``miku-on-desk-kiosk``）
决定，配置模型里不需要也不应该有一个"是否是 kiosk"的标志位。
"""

from __future__ import annotations

from pydantic import BaseModel


class KioskConfig(BaseModel):
    default_pet: str = "xff"
    character_scale: float = 2.5
    """精灵图放大倍数，让角色在 3.5 寸小屏幕上占据大部分画面——原始精灵帧是 128x128
    像素，放在竖屏 320x480 的画布里默认 1.0 倍会显得很小。"""
    rotate_90_clockwise: bool = True
    """MHS-3.5 屏幕现在用的 fbdev 驱动是固定横向 480x320，X11 层面报不出竖屏分辨率、
    也不支持 RandR 旋转/换模（``xrandr`` 只有一个固定 480x320 模式）——要在还没折腾
    Phase 0 的 ``dtoverlay=mhs35:rotate=`` 内核级旋转（会连带需要重新做触摸校准）之前
    先看到竖屏效果，只能在应用这一层把整个画面转过来。等 Phase 0 真的把屏幕转成物理
    竖屏、X11 直接报出竖屏分辨率之后，这里应该改回 ``False``，不需要再叠加一层旋转。
    """
