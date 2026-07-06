"""Hook sidecar：接收外部 CLI 工具（如 Claude Code）的 HTTP 通知，转发给上层回调。

零 Qt 依赖——``on_event`` 回调可以直接传入 ``HookEventBus.emit_event``
（见 ``bridge.py``），线程安全的原理与 ``bridge/events.py`` 的 ``BrainEventBus`` 完全
一致：Qt 的 AutoConnection 在跨线程 emit 时会自动升级为 QueuedConnection，本模块的
处理线程不需要自己维护线程安全队列，也不需要知道 Qt 的存在。

用 ``ThreadingHTTPServer`` 而非单线程 ``HTTPServer``：只是为了不让一次悬挂的慢请求
阻塞后续请求，不是为了应对高并发——hook 事件本身频率很低（每次工具调用一次）。

Token 每次进程启动都通过 ``token.rotate_token`` 重新生成并写入磁盘，构造
``HookServer`` 时即完成轮换，校验用 ``hmac.compare_digest`` 而非 ``==`` 以避免
基于响应时间差异的旁路猜测。
"""

from __future__ import annotations

import hmac
import json
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from miku_on_desk.face.hooks.schema import HookEvent
from miku_on_desk.face.hooks.token import rotate_token

PET_EVENT_PATH = "/pet-event"

_AUTH_PREFIX = "Bearer "


def _make_handler(
    on_event: Callable[[HookEvent], None], token: str
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass  # 静默：hook 事件高频且不需要访问日志噪音

        def do_POST(self) -> None:
            if self.path != PET_EVENT_PATH:
                self._respond(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if not self._authorized():
                self._respond(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self._respond(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
                return
            if not isinstance(payload, dict):
                self._respond(HTTPStatus.BAD_REQUEST, {"error": "expected a JSON object"})
                return

            on_event(HookEvent.from_raw(payload))
            self._respond(HTTPStatus.OK, {"status": "ok"})

        def _authorized(self) -> bool:
            header = self.headers.get("Authorization", "")
            if not header.startswith(_AUTH_PREFIX):
                return False
            return hmac.compare_digest(header[len(_AUTH_PREFIX) :], token)

        def _respond(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


class HookServer:
    """``port=0`` 让操作系统分配空闲端口（供测试用），真实运行时传固定端口。"""

    def __init__(
        self,
        on_event: Callable[[HookEvent], None],
        *,
        token_path: Path,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self._token = rotate_token(token_path)
        handler = _make_handler(on_event, self._token)
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._httpd.server_port

    @property
    def token(self) -> str:
        return self._token

    def start(self) -> None:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
