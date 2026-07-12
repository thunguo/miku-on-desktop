"""树莓派到 Pico USB HID 固件的最小确认协议。

Pico 固件应对每个 JSONL 指令回 ``{"seq": <n>, "ok": true}``。没有 Pico 或 ACK 不合法时
立即报错，调用方不会把“发送失败”伪装成目标主机已执行。
"""

from __future__ import annotations

import json
import select
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class HidTransportError(RuntimeError):
    pass


class HidTransport(Protocol):
    def click(self, x: int, y: int) -> None: ...

    def press_keys(self, keys: Sequence[str]) -> None: ...

    def type_text(self, text: str) -> None: ...


@dataclass
class UnavailableHidTransport:
    reason: str

    def _raise(self) -> None:
        raise HidTransportError(self.reason)

    def click(self, x: int, y: int) -> None:
        del x, y
        self._raise()

    def press_keys(self, keys: Sequence[str]) -> None:
        del keys
        self._raise()

    def type_text(self, text: str) -> None:
        del text
        self._raise()


@dataclass
class SerialHidTransport:
    device: Path
    timeout_s: float
    _next_sequence: int = field(default=1, init=False)

    def _send(self, payload: dict[str, Any]) -> None:
        if not self.device.exists():
            raise HidTransportError(f"未检测到 Pico HID 串口：{self.device}")
        sequence = self._next_sequence
        self._next_sequence += 1
        wire = json.dumps({"seq": sequence, **payload}, ensure_ascii=False).encode("utf-8") + b"\n"
        try:
            with self.device.open("r+b", buffering=0) as port:
                port.write(wire)
                ready, _, _ = select.select([port], [], [], self.timeout_s)
                if not ready:
                    raise HidTransportError("Pico HID 未在超时时间内确认指令")
                response = json.loads(port.readline().decode("utf-8"))
        except HidTransportError:
            raise
        except Exception as exc:
            raise HidTransportError(f"Pico HID 通讯失败：{exc}") from exc
        if response.get("seq") != sequence or response.get("ok") is not True:
            raise HidTransportError(f"Pico HID 拒绝指令：{response}")

    def click(self, x: int, y: int) -> None:
        self._send({"action": "click", "x": x, "y": y})

    def press_keys(self, keys: Sequence[str]) -> None:
        self._send({"action": "key_press", "keys": list(keys)})

    def type_text(self, text: str) -> None:
        self._send({"action": "type_text", "text": text})
