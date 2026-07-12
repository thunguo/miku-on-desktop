"""不依赖桌面会话的树莓派图像采集源。"""

from __future__ import annotations

import io
import shutil
import subprocess
from dataclasses import dataclass
from typing import Protocol

from PIL import Image, ImageChops, ImageStat

from miku_on_desk.hardware.device_config import CsiCameraConfig, HdmiCaptureConfig


class HardwareCaptureError(RuntimeError):
    """外设不存在、无信号或采集程序失败。"""


class FrameSource(Protocol):
    def capture(self) -> Image.Image: ...


class StillCameraSource(FrameSource, Protocol):
    def is_available(self) -> bool: ...

    def capture_png(self) -> bytes: ...


def _run_image_command(command: list[str], *, timeout_s: float, label: str) -> bytes:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        raise HardwareCaptureError(f"未安装 {label} 采集工具：{command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise HardwareCaptureError(f"{label} 采集超时") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise HardwareCaptureError(f"{label} 采集失败：{detail or exc.returncode}") from exc
    if not completed.stdout:
        raise HardwareCaptureError(f"{label} 未返回图像数据")
    return completed.stdout


def _decode_image(data: bytes, *, label: str) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as opened:
            return opened.convert("RGB")
    except Exception as exc:
        raise HardwareCaptureError(f"{label} 返回的不是有效图像") from exc


@dataclass(frozen=True)
class HdmiCaptureSource:
    config: HdmiCaptureConfig

    def capture(self) -> Image.Image:
        device = self.config.device
        if not device.exists():
            raise HardwareCaptureError(f"未检测到 HDMI 采集卡：{device}")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "v4l2",
            "-input_format",
            "mjpeg",
            "-video_size",
            f"{self.config.width}x{self.config.height}",
            "-framerate",
            str(self.config.fps),
            "-i",
            str(device),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        return _decode_image(
            _run_image_command(command, timeout_s=self.config.timeout_s, label="HDMI"),
            label="HDMI",
        )


@dataclass(frozen=True)
class RpiCameraStillSource:
    config: CsiCameraConfig

    def is_available(self) -> bool:
        return self.config.enabled and shutil.which(self.config.executable) is not None

    def capture_png(self) -> bytes:
        if not self.config.enabled:
            raise HardwareCaptureError("CSI 摄像头已在配置中关闭")
        command = [
            self.config.executable,
            "--nopreview",
            "--timeout",
            "700",
            "--width",
            str(self.config.still_width),
            "--height",
            str(self.config.still_height),
            "--encoding",
            "png",
            "--output",
            "-",
        ]
        return _run_image_command(command, timeout_s=self.config.timeout_s, label="CSI 摄像头")

    def capture(self) -> Image.Image:
        return _decode_image(self.capture_png(), label="CSI 摄像头")


@dataclass
class MotionDetector:
    """无模型的本地运动门控，避免在静止场景持续上传快照。"""

    previous: Image.Image | None = None
    edge: int = 96

    def changed(self, image: Image.Image, *, threshold: float) -> bool:
        current = image.convert("L").resize((self.edge, self.edge))
        if self.previous is None:
            self.previous = current
            return False
        difference = ImageChops.difference(self.previous, current)
        self.previous = current
        return float(ImageStat.Stat(difference).mean[0]) >= threshold


def build_hdmi_source(config: HdmiCaptureConfig) -> HdmiCaptureSource | None:
    return HdmiCaptureSource(config) if config.enabled else None


def build_csi_camera_source(config: CsiCameraConfig) -> RpiCameraStillSource | None:
    return RpiCameraStillSource(config) if config.enabled else None
