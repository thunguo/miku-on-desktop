"""SSE / Streamable HTTP transport 集成测试用的真实子进程 fixture server 管理：分配临时
端口、轮询等待启动完成、测试结束后清理子进程——重试/轮询逻辑只存在于测试代码里，不进
生产的 `client.py`（`client.py` 的 docstring 明确不做自动重连）。
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

_FIXTURE_SERVER = Path(__file__).parent / "_fixture_server.py"


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def spawn_fixture_server(transport: str) -> Iterator[tuple[str, int]]:
    port = _free_tcp_port()
    proc = subprocess.Popen(
        [sys.executable, str(_FIXTURE_SERVER), "--transport", transport, "--port", str(port)]
    )
    try:
        _wait_until_ready(port, transport)
        yield "127.0.0.1", port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _wait_until_ready(port: int, transport: str, timeout: float = 5.0) -> None:
    # SSE 的 /sse 端点是长连接流，`httpx.get()` 会等body读完才返回，导致 ReadTimeout；
    # 用 `httpx.stream()` 只等响应头即可判断服务是否就绪，不进入 body 迭代。
    path = "/sse" if transport == "sse" else "/mcp"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with httpx.stream("GET", f"http://127.0.0.1:{port}{path}", timeout=0.2):
                return
        except httpx.ConnectError:
            time.sleep(0.05)
    raise TimeoutError(f"fixture server 在 {timeout}s 内未就绪（transport={transport}）")
