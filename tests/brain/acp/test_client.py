"""run_acp_task 对真实 ACP 子进程（`_fixture_agent.py`）的回归测试。

不 mock ClientSideConnection——直接拉起 `_fixture_agent.py` 作为子进程，走完整的官方
`agent-client-protocol` SDK 握手/prompt/权限请求流程，验证的是"这套封装真的能对上 SDK 的
线上协议行为"。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from miku_on_desk.brain.acp.client import run_acp_task
from miku_on_desk.brain.tools.path_sandbox import PathSandbox

_FIXTURE_AGENT = Path(__file__).parent / "_fixture_agent.py"


async def test_run_acp_task_returns_agent_message_on_success(tmp_path: Path) -> None:
    result = await run_acp_task(
        executable=sys.executable, args=[str(_FIXTURE_AGENT)], cwd=str(tmp_path), task="echo:你好"
    )

    assert result.success is True
    assert result.content == "你好"
    assert result.stop_reason == "end_turn"
    assert result.error is None


async def test_run_acp_task_reports_non_end_turn_stop_reason_as_failure(tmp_path: Path) -> None:
    result = await run_acp_task(
        executable=sys.executable, args=[str(_FIXTURE_AGENT)], cwd=str(tmp_path), task="refuse"
    )

    assert result.success is False
    assert result.stop_reason == "refusal"
    assert result.error is not None


async def test_run_acp_task_auto_approves_permission_requests(tmp_path: Path) -> None:
    result = await run_acp_task(
        executable=sys.executable,
        args=[str(_FIXTURE_AGENT)],
        cwd=str(tmp_path),
        task="request_permission",
    )

    assert result.success is True
    assert result.content == "approved:allow-option"


async def test_run_acp_task_times_out_and_reports_error(tmp_path: Path) -> None:
    result = await run_acp_task(
        executable=sys.executable,
        args=[str(_FIXTURE_AGENT)],
        cwd=str(tmp_path),
        task="sleep:5",
        timeout_s=0.2,
    )

    assert result.success is False
    assert result.error is not None
    assert "超时" in result.error


async def test_run_acp_task_with_bad_executable_returns_failure() -> None:
    result = await run_acp_task(
        executable="this-binary-does-not-exist-anywhere",
        cwd=".",
        task="echo:x",
        max_retries=0,
    )

    assert result.success is False
    assert result.error is not None


async def test_run_acp_task_invokes_on_chunk_for_each_streamed_fragment(tmp_path: Path) -> None:
    received: list[str] = []

    result = await run_acp_task(
        executable=sys.executable,
        args=[str(_FIXTURE_AGENT)],
        cwd=str(tmp_path),
        task="echo_multi:a|b|c",
        on_chunk=received.append,
    )

    assert result.success is True
    assert result.content == "abc"
    assert received == ["a", "b", "c"]


async def _fake_sleep(_delay: float) -> None:
    return None


async def test_run_acp_task_retries_handshake_after_transient_failure(tmp_path: Path) -> None:
    counter_file = tmp_path / "fail_count"
    counter_file.write_text("2")
    sleep_calls: list[float] = []

    async def _tracking_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    result = await run_acp_task(
        executable=sys.executable,
        args=[str(_FIXTURE_AGENT)],
        cwd=str(tmp_path),
        task="echo:你好",
        env={"FIXTURE_FAIL_COUNT_FILE": str(counter_file)},
        max_retries=2,
        sleep=_tracking_sleep,
    )

    assert result.success is True
    assert result.content == "你好"
    assert len(sleep_calls) == 2


async def test_run_acp_task_gives_up_after_max_handshake_retries(tmp_path: Path) -> None:
    counter_file = tmp_path / "fail_count"
    counter_file.write_text("100")

    result = await run_acp_task(
        executable=sys.executable,
        args=[str(_FIXTURE_AGENT)],
        cwd=str(tmp_path),
        task="echo:你好",
        env={"FIXTURE_FAIL_COUNT_FILE": str(counter_file)},
        max_retries=2,
        sleep=_fake_sleep,
    )

    assert result.success is False
    assert result.error is not None
    assert int(counter_file.read_text()) == 97


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PathSandbox:
    # 与 test_policy.py 中的理由相同：隔离系统临时目录，避免 tmp_path 落在其下导致
    # "outside" 类测试路径被临时目录规则误判为允许。
    monkeypatch.setattr(
        "miku_on_desk.brain.tools.path_sandbox.tempfile.gettempdir",
        lambda: str(tmp_path / "system_tmp"),
    )
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    return PathSandbox(cwd=cwd, output_dir=tmp_path / "output", data_dir=tmp_path / "data")


async def test_write_text_file_without_sandbox_is_unrestricted(tmp_path: Path) -> None:
    """不传 `path_sandbox`（默认 None）时行为与接入前一致，验证接入不改变默认路径。"""
    target = tmp_path / "outside" / "note.txt"
    target.parent.mkdir()

    result = await run_acp_task(
        executable=sys.executable,
        args=[str(_FIXTURE_AGENT)],
        cwd=str(tmp_path),
        task=f"write_file:{target}|hello",
    )

    assert result.content == "write_ok"
    assert target.read_text() == "hello"


async def test_write_text_file_inside_sandbox_succeeds(
    sandbox: PathSandbox, tmp_path: Path
) -> None:
    cwd = tmp_path / "cwd"
    target = cwd / "note.txt"

    result = await run_acp_task(
        executable=sys.executable,
        args=[str(_FIXTURE_AGENT)],
        cwd=str(cwd),
        task=f"write_file:{target}|hello",
        path_sandbox=sandbox,
    )

    assert result.content == "write_ok"
    assert target.read_text() == "hello"


async def test_write_text_file_outside_sandbox_is_denied(
    sandbox: PathSandbox, tmp_path: Path
) -> None:
    cwd = tmp_path / "cwd"
    outside = tmp_path / "outside" / "note.txt"

    result = await run_acp_task(
        executable=sys.executable,
        args=[str(_FIXTURE_AGENT)],
        cwd=str(cwd),
        task=f"write_file:{outside}|hello",
        path_sandbox=sandbox,
    )

    assert result.content.startswith("write_denied:")
    assert not outside.exists()


async def test_read_text_file_outside_sandbox_is_denied(
    sandbox: PathSandbox, tmp_path: Path
) -> None:
    cwd = tmp_path / "cwd"
    outside = tmp_path / "outside" / "secret.txt"
    outside.parent.mkdir()
    outside.write_text("top secret")

    result = await run_acp_task(
        executable=sys.executable,
        args=[str(_FIXTURE_AGENT)],
        cwd=str(cwd),
        task=f"read_file:{outside}",
        path_sandbox=sandbox,
    )

    assert result.content.startswith("read_denied:")
    assert "top secret" not in result.content

