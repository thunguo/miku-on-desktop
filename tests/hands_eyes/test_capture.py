"""capture.py 的回归测试：假 mss.MSS 返回的显示器/截图数据，不实际截屏。"""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from miku_on_desk.hands_eyes import capture as capture_module
from miku_on_desk.hands_eyes.capture import capture_origin, capture_screen, crop_to_bounds


class _FakeShot:
    def __init__(self, size: tuple[int, int], rgb: bytes) -> None:
        self.size = size
        self.rgb = rgb


class _FakeSct:
    def __init__(self, monitors: list[dict[str, Any]], shot: _FakeShot | None = None) -> None:
        self.monitors = monitors
        self._shot = shot

    def grab(self, monitor: dict[str, Any]) -> _FakeShot:
        assert self._shot is not None
        return self._shot

    def __enter__(self) -> _FakeSct:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_capture_screen_grabs_monitors_zero_and_wraps_as_pil_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_shot = _FakeShot(size=(2, 2), rgb=bytes(2 * 2 * 3))
    monitors = [{"left": -100, "top": 0, "width": 2, "height": 2}]
    fake_sct = _FakeSct(monitors=monitors, shot=fake_shot)
    monkeypatch.setattr(capture_module.mss, "MSS", lambda: fake_sct)

    image = capture_screen()

    assert image.size == (2, 2)


def test_capture_origin_returns_monitors_zero_left_top(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sct = _FakeSct(monitors=[{"left": -100, "top": 50, "width": 800, "height": 600}])
    monkeypatch.setattr(capture_module.mss, "MSS", lambda: fake_sct)

    assert capture_origin() == (-100, 50)


def test_crop_to_bounds_crops_and_returns_global_offset() -> None:
    image = Image.new("RGB", (200, 150))

    cropped, offset = crop_to_bounds(image, bounds=(1050, 2030, 40, 20), origin=(1000, 2000))

    assert cropped.size == (40, 20)
    assert offset == (1050, 2030)


def test_crop_to_bounds_clamps_bounds_partially_outside_image() -> None:
    image = Image.new("RGB", (100, 100))

    cropped, offset = crop_to_bounds(image, bounds=(80, 80, 50, 50), origin=(0, 0))

    assert cropped.size == (20, 20)
    assert offset == (80, 80)


def test_crop_to_bounds_returns_original_image_and_origin_when_bounds_fully_outside() -> None:
    image = Image.new("RGB", (100, 100))

    cropped, offset = crop_to_bounds(image, bounds=(500, 500, 10, 10), origin=(7, 9))

    assert cropped is image
    assert offset == (7, 9)
