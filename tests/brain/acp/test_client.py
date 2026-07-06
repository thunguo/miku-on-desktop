"""run_acp_task 对真实 ACP 子进程（`_fixture_agent.py`）的回归测试。

不 mock ClientSideConnection——直接拉起 `_fixture_agent.py` 作为子进程，走完整的官方
`agent-client-protocol` SDK 握手/prompt/权限请求流程，验证的是"这套封装真的能对上 SDK 的
线上协议行为"。
"""

from __future__ import annotations

import sys
from pathlib import Path

from miku_on_desk.brain.acp.client import run_acp_task

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


async def test_run_acp_task_with_bad_executable_returns_failure(tmp_path: Path) -> None:
    result = await run_acp_task(
        executable="this-binary-does-not-exist-anywhere", cwd=str(tmp_path), task="echo:x"
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
