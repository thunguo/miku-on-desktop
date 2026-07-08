"""MCPServerConnection 对真实 stdio 子进程（`_fixture_server.py`）的回归测试。

不 mock ClientSession/transport——直接拉起 `_fixture_server.py` 作为子进程，走完整的官方
`mcp` SDK 握手/工具发现/调用流程，验证的是"这套封装真的能对上 SDK 的线上协议行为"。
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import pytest

from miku_on_desk.brain.mcp.client import ConnectionState, MCPServerConnection, MCPToolError
from miku_on_desk.config.settings import McpServerConfig

_FIXTURE_SERVER = Path(__file__).parent / "_fixture_server.py"


def _fixture_config() -> McpServerConfig:
    return McpServerConfig(name="fixture", command=sys.executable, args=[str(_FIXTURE_SERVER)])


@pytest.fixture
async def connection() -> MCPServerConnection:
    conn = MCPServerConnection(_fixture_config())
    yield conn
    await conn.disconnect()


async def test_connect_discovers_tools_and_reaches_connected_state(
    connection: MCPServerConnection,
) -> None:
    await connection.connect()

    assert connection.state == ConnectionState.CONNECTED
    assert {tool.name for tool in connection.tools} == {"echo", "fail"}


async def test_call_tool_returns_flattened_text_content(connection: MCPServerConnection) -> None:
    await connection.connect()

    result = await connection.call_tool("echo", {"text": "你好"})

    assert result == "你好"


async def test_call_tool_raises_mcp_tool_error_on_server_side_failure(
    connection: MCPServerConnection,
) -> None:
    await connection.connect()

    with pytest.raises(MCPToolError):
        await connection.call_tool("fail", {"reason": "故意失败"})


async def test_call_tool_before_connect_raises_mcp_tool_error(
    connection: MCPServerConnection,
) -> None:
    with pytest.raises(MCPToolError):
        await connection.call_tool("echo", {"text": "x"})


async def test_disconnect_resets_state_to_disconnected(connection: MCPServerConnection) -> None:
    await connection.connect()

    await connection.disconnect()

    assert connection.state == ConnectionState.DISCONNECTED


async def test_status_reports_tool_count(connection: MCPServerConnection) -> None:
    await connection.connect()

    status = connection.status()

    assert status.name == "fixture"
    assert status.state == ConnectionState.CONNECTED
    assert status.tool_count == 2
    assert status.error_message is None


async def test_connect_with_bad_command_sets_error_state() -> None:
    bad_connection = MCPServerConnection(
        McpServerConfig(name="broken", command="this-binary-does-not-exist-anywhere")
    )

    with pytest.raises(Exception):  # noqa: B017 - 具体异常类型由 subprocess 创建失败决定
        await bad_connection.connect()

    assert bad_connection.state == ConnectionState.ERROR
    assert bad_connection.status().error_message is not None


async def test_reconnect_restores_connected_state(connection: MCPServerConnection) -> None:
    await connection.connect()

    await connection.reconnect()

    assert connection.state == ConnectionState.CONNECTED
    assert {tool.name for tool in connection.tools} == {"echo", "fail"}


async def test_watchdog_self_heals_after_ping_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """看门狗周期性探活，探活失败后自愈重连——用一个真实的新子进程换掉旧连接。"""
    conn = MCPServerConnection(_fixture_config(), watchdog_interval_s=0.05)
    try:
        await conn.connect()
        stale_session = conn._session
        assert stale_session is not None

        async def _failing_ping(**kwargs: object) -> None:
            raise RuntimeError("模拟探活失败")

        monkeypatch.setattr(stale_session, "send_ping", _failing_ping)

        async def _wait_for_recovery() -> None:
            while conn._session is stale_session or conn.state != ConnectionState.CONNECTED:
                await asyncio.sleep(0.02)

        await asyncio.wait_for(_wait_for_recovery(), timeout=5.0)

        assert conn.status().reconnect_attempts == 0
        assert {tool.name for tool in conn.tools} == {"echo", "fail"}
    finally:
        await conn.disconnect()


async def test_self_heal_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    conn = MCPServerConnection(_fixture_config(), max_reconnect_retries=2, sleep=_fake_sleep)
    try:
        await conn.connect()
        # 这里只测自愈循环自身的重试/放弃逻辑，不需要看门狗参与；而看门狗任务和自愈共用同一个
        # 被注入的空操作 sleep，放着不停会在 while True 里连一次真实挂起都没有地忙等，饿死事件循环。
        await conn._stop_watchdog()

        async def _always_fail(*args: object, **kwargs: object) -> object:
            raise RuntimeError("模拟连接彻底失败")

        monkeypatch.setattr(conn, "_open_transport", _always_fail)

        await conn._self_heal()

        assert conn.state == ConnectionState.ERROR
        assert conn.status().reconnect_attempts == 2
        assert len(sleep_calls) == 2
    finally:
        await conn.disconnect()


async def test_self_heal_is_singleflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """并发触发两次自愈，实际只应该发起一次真实重连（单飞锁 + 状态 guard）。"""
    call_count = 0
    conn = MCPServerConnection(_fixture_config())
    try:
        await conn.connect()
        real_open_transport = conn._open_transport

        async def _counting_open_transport(exit_stack: AsyncExitStack) -> Any:
            nonlocal call_count
            call_count += 1
            return await real_open_transport(exit_stack)

        monkeypatch.setattr(conn, "_open_transport", _counting_open_transport)

        await asyncio.gather(conn._self_heal(), conn._self_heal())

        assert call_count == 1
        assert conn.state == ConnectionState.CONNECTED
    finally:
        await conn.disconnect()
