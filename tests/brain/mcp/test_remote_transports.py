"""SSE / Streamable HTTP transport 的真实集成测试：起一个真正的子进程 fixture server（见
``_http_fixture.py``），走完整的 MCP 握手/工具发现/调用流程——不 mock transport 层，与现有
stdio 测试（`test_client.py`/`test_host.py`）同样的验证哲学。

两种远程 transport 共享大部分断言逻辑，用 ``@pytest.mark.parametrize`` 合并，减少重复；
header 透传单独用 ``echo_header`` 工具验证一次，确认 `McpServerConfig.headers` 里配置的自定义
HTTP header 确实被服务端收到（而不仅仅是 client 端"发送了但没人读"）。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from miku_on_desk.brain.mcp.client import ConnectionState, MCPServerConnection, MCPToolError
from miku_on_desk.config.settings import McpServerConfig, McpTransport
from tests.brain.mcp._http_fixture import spawn_fixture_server

_REMOTE_TRANSPORTS = [McpTransport.SSE, McpTransport.STREAMABLE_HTTP]


@pytest.fixture(params=_REMOTE_TRANSPORTS)
def remote_server(request: pytest.FixtureRequest) -> Iterator[tuple[McpTransport, str, int]]:
    transport: McpTransport = request.param
    with spawn_fixture_server(transport.value) as (host, port):
        yield transport, host, port


def _config(
    transport: McpTransport, host: str, port: int, headers: dict[str, str] | None = None
) -> McpServerConfig:
    path = "sse" if transport is McpTransport.SSE else "mcp"
    return McpServerConfig(
        name="remote-fixture",
        transport=transport,
        url=f"http://{host}:{port}/{path}",
        headers=headers or {},
    )


async def test_connect_discovers_tools_and_reaches_connected_state(
    remote_server: tuple[McpTransport, str, int],
) -> None:
    transport, host, port = remote_server
    connection = MCPServerConnection(_config(transport, host, port))

    await connection.connect()
    try:
        assert connection.state == ConnectionState.CONNECTED
        assert {tool.name for tool in connection.tools} == {"echo", "fail", "echo_header"}
    finally:
        await connection.disconnect()


async def test_call_tool_returns_flattened_text_content(
    remote_server: tuple[McpTransport, str, int],
) -> None:
    transport, host, port = remote_server
    connection = MCPServerConnection(_config(transport, host, port))
    await connection.connect()

    try:
        result = await connection.call_tool("echo", {"text": "你好"})
        assert result == "你好"
    finally:
        await connection.disconnect()


async def test_call_tool_raises_mcp_tool_error_on_server_side_failure(
    remote_server: tuple[McpTransport, str, int],
) -> None:
    transport, host, port = remote_server
    connection = MCPServerConnection(_config(transport, host, port))
    await connection.connect()

    try:
        with pytest.raises(MCPToolError):
            await connection.call_tool("fail", {"reason": "故意失败"})
    finally:
        await connection.disconnect()


async def test_disconnect_resets_state_to_disconnected(
    remote_server: tuple[McpTransport, str, int],
) -> None:
    transport, host, port = remote_server
    connection = MCPServerConnection(_config(transport, host, port))
    await connection.connect()

    await connection.disconnect()

    assert connection.state == ConnectionState.DISCONNECTED


async def test_custom_header_is_received_by_the_server(
    remote_server: tuple[McpTransport, str, int],
) -> None:
    transport, host, port = remote_server
    connection = MCPServerConnection(
        _config(transport, host, port, headers={"X-Test-Header": "hello123"})
    )
    await connection.connect()

    try:
        result = await connection.call_tool("echo_header", {"header_name": "x-test-header"})
        assert result == "hello123"
    finally:
        await connection.disconnect()
