"""``face/hooks/forward.py`` 的回归测试：``_with_source`` 的 payload 改写逻辑，以及
``main()`` 在网络失败时仍返回退出码 0、不往 stdout 写任何内容。
"""

from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import ClassVar

import pytest

from miku_on_desk.face.hooks.forward import _with_source, main


def test_with_source_adds_source_field_to_valid_json() -> None:
    body = _with_source('{"event": "Stop"}', "codex")

    assert json.loads(body) == {"event": "Stop", "source": "codex"}


def test_with_source_does_not_override_existing_source_field() -> None:
    body = _with_source('{"event": "Stop", "source": "already_set"}', "codex")

    assert json.loads(body) == {"event": "Stop", "source": "already_set"}


def test_with_source_passes_through_invalid_json_unchanged() -> None:
    body = _with_source("not json", "codex")

    assert body == b"not json"


def test_with_source_passes_through_non_dict_json_unchanged() -> None:
    body = _with_source("[1, 2, 3]", "codex")

    assert body == b"[1, 2, 3]"


def test_with_source_handles_empty_stdin() -> None:
    body = _with_source("", "codex")

    assert json.loads(body) == {"source": "codex"}


def test_main_returns_zero_and_writes_nothing_to_stdout_when_target_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO('{"event": "Stop"}'))
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)

    exit_code = main(
        ["--url", "http://127.0.0.1:1/pet-event", "--token", "tok1", "--source", "codex"]
    )

    assert exit_code == 0
    assert stdout.getvalue() == ""


class _RecordingHandler(BaseHTTPRequestHandler):
    received_bodies: ClassVar[list[bytes]] = []
    received_headers: ClassVar[list[dict[str, str]]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.received_bodies.append(self.rfile.read(length))
        self.received_headers.append(dict(self.headers))
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_main_forwards_stdin_payload_with_auth_header_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordingHandler.received_bodies = []
    _RecordingHandler.received_headers = []
    server = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        monkeypatch.setattr("sys.stdin", io.StringIO('{"event": "Stop"}'))
        url = f"http://127.0.0.1:{server.server_port}/pet-event"

        exit_code = main(["--url", url, "--token", "tok1", "--source", "gemini_cli"])

        assert exit_code == 0
        assert len(_RecordingHandler.received_bodies) == 1
        body = json.loads(_RecordingHandler.received_bodies[0])
        assert body == {"event": "Stop", "source": "gemini_cli"}
        assert _RecordingHandler.received_headers[0]["Authorization"] == "Bearer tok1"
    finally:
        server.shutdown()
        thread.join()
