"""MCPServerConnection 对真实 stdio 子进程（`_fixture_server.py`）的回归测试。

不 mock ClientSession/transport——直接拉起 `_fixture_server.py` 作为子进程，走完整的官方
`mcp` SDK 握手/工具发现/调用流程，验证的是"这套封装真的能对上 SDK 的线上协议行为"。
"""

from __future__ import annotations

import sys
from pathlib import Path

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
