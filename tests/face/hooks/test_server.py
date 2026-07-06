"""server.py 的回归测试：真实 HTTP 往返（``urllib.request``，仓库未引入 ``requests``/
``httpx``），覆盖鉴权、JSON 校验、路径匹配与事件回调分发。

每个测试自己起一个 ``port=0``（操作系统分配空闲端口）的 ``HookServer``，用完在 finally
里 ``stop()``，避免端口/线程泄漏到下一个测试。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from miku_on_desk.face.hooks.schema import HookEvent
from miku_on_desk.face.hooks.server import PET_EVENT_PATH, HookServer


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[HookServer, list[HookEvent]]]:
    received: list[HookEvent] = []
    srv = HookServer(received.append, token_path=tmp_path / "hook_token", port=0)
    srv.start()
    try:
        yield srv, received
    finally:
        srv.stop()


def _post(
    srv: HookServer, *, path: str = PET_EVENT_PATH, token: str | None = None, body: bytes
) -> tuple[int, dict[str, object]]:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"http://127.0.0.1:{srv.port}{path}", data=body, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_port_is_assigned_by_os_and_usable(server: tuple[HookServer, list[HookEvent]]) -> None:
    srv, _ = server

    assert srv.port != 0


def test_valid_token_returns_200_and_invokes_callback(
    server: tuple[HookServer, list[HookEvent]],
) -> None:
    srv, received = server
    payload = {"event": "PostToolUseFailure", "tool_name": "Bash", "error": "boom"}

    status, body = _post(srv, token=srv.token, body=json.dumps(payload).encode())

    assert status == 200
    assert body == {"status": "ok"}
    assert len(received) == 1
    assert received[0].event == "PostToolUseFailure"
    assert received[0].tool_name == "Bash"
    assert received[0].error == "boom"


def test_missing_token_returns_401_and_does_not_invoke_callback(
    server: tuple[HookServer, list[HookEvent]],
) -> None:
    srv, received = server

    status, body = _post(srv, body=json.dumps({"event": "SessionStart"}).encode())

    assert status == 401
    assert body == {"error": "unauthorized"}
    assert received == []


def test_wrong_token_returns_401_and_does_not_invoke_callback(
    server: tuple[HookServer, list[HookEvent]],
) -> None:
    srv, received = server

    status, _ = _post(
        srv, token="not-the-real-token", body=json.dumps({"event": "SessionStart"}).encode()
    )

    assert status == 401
    assert received == []


def test_malformed_json_body_returns_400(
    server: tuple[HookServer, list[HookEvent]],
) -> None:
    srv, received = server

    status, body = _post(srv, token=srv.token, body=b"{not valid json")

    assert status == 400
    assert body == {"error": "invalid JSON"}
    assert received == []


def test_non_object_json_body_returns_400(
    server: tuple[HookServer, list[HookEvent]],
) -> None:
    srv, received = server

    status, body = _post(srv, token=srv.token, body=json.dumps(["a", "list"]).encode())

    assert status == 400
    assert body == {"error": "expected a JSON object"}
    assert received == []


def test_wrong_path_returns_404(server: tuple[HookServer, list[HookEvent]]) -> None:
    srv, received = server

    status, body = _post(srv, path="/nope", token=srv.token, body=b"{}")

    assert status == 404
    assert body == {"error": "not found"}
    assert received == []
