"""树莓派直渲染 kiosk 的启动失败提示回归测试。"""

from __future__ import annotations

from miku_on_desk.kiosk_main import _startup_failure_message


def test_startup_failure_message_keeps_details_in_journal() -> None:
    message = _startup_failure_message("KIOSK-AB12CD34")

    assert "KIOSK-AB12CD34" in message
    assert "journalctl -u miku-kiosk.service" in message
    assert "Traceback" not in message
