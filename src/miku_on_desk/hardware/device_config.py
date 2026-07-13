"""树莓派外设配置。

设备路径必须优先使用 udev 的稳定链接，而不是 ``/dev/videoN``：CSI 摄像头、HDMI 采集卡
的枚举顺序会随开机时序改变。所有主动视觉默认关闭，避免设备首次启动便开始采集用户画面。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class HdmiCaptureConfig(BaseModel):
    enabled: bool = False
    device: Path = Path("/dev/v4l/by-id/usb-MACROSILICON_USB_Video-video-index0")
    width: int = Field(default=1920, ge=16)
    height: int = Field(default=1080, ge=16)
    fps: int = Field(default=30, ge=1, le=60)
    timeout_s: float = Field(default=8.0, gt=0, le=30)


class CsiCameraConfig(BaseModel):
    enabled: bool = True
    still_width: int = Field(default=640, ge=16)
    still_height: int = Field(default=480, ge=16)
    timeout_s: float = Field(default=12.0, gt=0, le=30)
    executable: str = "rpicam-still"


class PresenceCameraConfig(BaseModel):
    """摄像头在场观察。

    本地只用缩小灰度帧差筛掉静止画面；发生运动后才把单张快照交给视觉模型确认是否有人。
    图像只保存在内存中，不写文件，也不用来识别身份。
    """

    enabled: bool = False
    scan_interval_s: int = Field(default=20, ge=5, le=3600)
    min_trigger_interval_s: int = Field(default=900, ge=60, le=86_400)
    motion_threshold: float = Field(default=8.0, ge=0.1, le=255.0)


class HidConfig(BaseModel):
    enabled: bool = False
    serial_device: Path | None = None
    timeout_s: float = Field(default=1.5, gt=0, le=10)


class HardwareConfig(BaseModel):
    hdmi: HdmiCaptureConfig = Field(default_factory=HdmiCaptureConfig)
    csi_camera: CsiCameraConfig = Field(default_factory=CsiCameraConfig)
    presence_camera: PresenceCameraConfig = Field(default_factory=PresenceCameraConfig)
    hid: HidConfig = Field(default_factory=HidConfig)
