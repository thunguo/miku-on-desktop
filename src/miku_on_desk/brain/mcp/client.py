"""单个 MCP server 连接封装。

用官方 `mcp` Python SDK（`mcp>=1.2`，已在 pyproject.toml 声明）承担协议层——SDK 自带
`stdio_client` transport 与 `ClientSession` 的握手/工具发现/调用封装，不需要自己实现
JSON-RPC 2.0 帧协议和请求 id、超时、通知分发这些细节。

支持三种 transport：`stdio`（本机子进程，command/args/env）、`sse`/`streamable-http`
（远程 MCP server，url/headers，headers 通常用于鉴权）。`_open_transport()` 按
`McpServerConfig.transport` 分派，三者产出的 `(read, write)` 流对象形状一致，`connect()`
之后的握手/工具发现逻辑不需要感知具体是哪种 transport。

不做 transport-close 时的自动指数退避重连：这依赖对 SDK 的 stdio 子进程生命周期做主动监听
才能可靠判断"意外断开"，在没有真实崩溃场景可验证的前提下容易做出一套自己也不确定生效的重连
逻辑。这里只保留 `reconnect()` 这个显式方法，交给上层（未来的设置面板"重连"按钮）决定何时
调用，不做后台自动重连。同理，不监听服务端 `notifications/tools/list_changed` 主动通知来
热更新工具列表——需要时可以再补，热更新与手动 `reconnect()` 都会重新跑 `tools/list`,效果等价。
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.message import SessionMessage

from miku_on_desk.config.settings import McpServerConfig, McpTransport

logger = logging.getLogger(__name__)


class ConnectionState(StrEnum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class MCPToolError(Exception):
    """`tools/call` 返回 ``isError=True`` 时抛出，携带服务端给出的错误内容文本。"""


@dataclass(frozen=True)
class MCPServerStatus:
    name: str
    state: ConnectionState
    tool_count: int
    error_message: str | None = None


def _flatten_content(content: list[types.ContentBlock]) -> str:
    parts: list[str] = []
    for block in content:
        if isinstance(block, types.TextContent):
            parts.append(block.text)
        elif isinstance(block, types.ImageContent):
            parts.append(f"[Image: {block.mimeType}]")
        elif isinstance(block, types.AudioContent):
            parts.append(f"[Audio: {block.mimeType}]")
        elif isinstance(block, types.EmbeddedResource):
            parts.append(f"[Resource: {block.resource.uri}]")
        elif isinstance(block, types.ResourceLink):
            parts.append(f"[Resource: {block.uri}]")
    return "\n".join(parts)


class MCPServerConnection:
    """单个 MCP server 的连接、握手、工具发现与调用。"""

    def __init__(self, config: McpServerConfig) -> None:
        self._config = config
        self._state = ConnectionState.DISCONNECTED
        self._error_message: str | None = None
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools: list[types.Tool] = []

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def tools(self) -> list[types.Tool]:
        return list(self._tools)

    def status(self) -> MCPServerStatus:
        return MCPServerStatus(
            name=self._config.name,
            state=self._state,
            tool_count=len(self._tools),
            error_message=self._error_message,
        )

    async def connect(self) -> None:
        if self._state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
            return
        self._state = ConnectionState.CONNECTING
        try:
            exit_stack = AsyncExitStack()
            read, write = await self._open_transport(exit_stack)
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._exit_stack = exit_stack
            self._session = session
            await self._discover_tools()
            self._state = ConnectionState.CONNECTED
            self._error_message = None
            logger.info(
                'MCP server "%s" 已连接，发现 %d 个工具', self._config.name, len(self._tools)
            )
        except Exception as exc:
            self._state = ConnectionState.ERROR
            self._error_message = str(exc)
            await self._cleanup()
            raise

    async def disconnect(self) -> None:
        await self._cleanup()
        self._state = ConnectionState.DISCONNECTED
        self._error_message = None

    async def reconnect(self) -> None:
        await self._cleanup()
        await self.connect()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if self._session is None or self._state != ConnectionState.CONNECTED:
            raise MCPToolError(f'MCP server "{self._config.name}" 未连接。')
        result = await self._session.call_tool(tool_name, arguments)
        text = _flatten_content(result.content)
        if result.isError:
            raise MCPToolError(text or f'工具 "{tool_name}" 执行失败。')
        return text

    async def _open_transport(
        self, exit_stack: AsyncExitStack
    ) -> tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
    ]:
        config = self._config
        if config.transport is McpTransport.STDIO:
            assert config.command is not None
            params = StdioServerParameters(
                command=config.command, args=config.args, env=config.env or None
            )
            return await exit_stack.enter_async_context(stdio_client(params))
        if config.transport is McpTransport.SSE:
            assert config.url is not None
            read, write = await exit_stack.enter_async_context(
                sse_client(config.url, headers=config.headers or None)
            )
            return read, write
        assert config.url is not None
        http_client = await exit_stack.enter_async_context(
            create_mcp_http_client(headers=config.headers or None)
        )
        read, write, _get_session_id = await exit_stack.enter_async_context(
            streamable_http_client(config.url, http_client=http_client)
        )
        return read, write

    async def _discover_tools(self) -> None:
        assert self._session is not None
        try:
            result = await self._session.list_tools()
            self._tools = list(result.tools)
        except Exception:
            logger.warning('MCP server "%s" 不支持 tools/list', self._config.name, exc_info=True)
            self._tools = []

    async def _cleanup(self) -> None:
        self._session = None
        if self._exit_stack is not None:
            exit_stack = self._exit_stack
            self._exit_stack = None
            try:
                await exit_stack.aclose()
            except Exception:
                logger.warning('MCP server "%s" 关闭连接时出错', self._config.name, exc_info=True)
