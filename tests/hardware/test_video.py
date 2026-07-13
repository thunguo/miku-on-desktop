"""树莓派 HDMI/CSI 图像源：命令构造、解码与本地运动门控。"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from miku_on_desk.hardware.device_config import CsiCameraConfig, HdmiCaptureConfig
from miku_on_desk.hardware.video import (
    HardwareCaptureError,
    HdmiCaptureSource,
    MotionDetector,
    RpiCameraStillSource,
)


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_hdmi_source_uses_configured_stable_video_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = tmp_path / "capture"
    device.touch()
    calls: list[list[str]] = []

    def _run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=_png_bytes((1, 2, 3)))

    monkeypatch.setattr("miku_on_desk.hardware.video.subprocess.run", _run)
    source = HdmiCaptureSource(HdmiCaptureConfig(enabled=True, device=device))

    image = source.capture()

    assert image.size == (4, 3)
    assert calls[0][calls[0].index("-i") + 1] == str(device)
    assert "mjpeg" in calls[0]


def test_hdmi_source_rejects_missing_device(tmp_path: Path) -> None:
    source = HdmiCaptureSource(HdmiCaptureConfig(enabled=True, device=tmp_path / "missing"))

    with pytest.raises(HardwareCaptureError, match="未检测到 HDMI 采集卡"):
        source.capture()


def test_csi_camera_source_emits_png_from_rpicam(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "miku_on_desk.hardware.video.shutil.which", lambda _name: "/usr/bin/rpicam-still"
    )

    def _run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=_png_bytes((4, 5, 6)))

    monkeypatch.setattr("miku_on_desk.hardware.video.subprocess.run", _run)
    source = RpiCameraStillSource(CsiCameraConfig())

    assert source.is_available() is True
    assert source.capture().getpixel((0, 0)) == (4, 5, 6)
    assert calls[0][0] == "rpicam-still"
    assert "--output" in calls[0]


def test_motion_detector_ignores_initial_and_static_frame_then_detects_change() -> None:
    detector = MotionDetector(edge=4)
    dark = Image.new("RGB", (8, 8), (0, 0, 0))
    bright = Image.new("RGB", (8, 8), (255, 255, 255))

    assert detector.changed(dark, threshold=2) is False
    assert detector.changed(dark, threshold=2) is False
    assert detector.changed(bright, threshold=2) is True
