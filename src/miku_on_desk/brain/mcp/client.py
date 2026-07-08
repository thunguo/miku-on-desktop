"""单个 MCP server 连接封装。

用官方 `mcp` Python SDK（`mcp>=1.2`，已在 pyproject.toml 声明）承担协议层——SDK 自带
`stdio_client` transport 与 `ClientSession` 的握手/工具发现/调用封装，不需要自己实现
JSON-RPC 2.0 帧协议和请求 id、超时、通知分发这些细节。

支持三种 transport：`stdio`（本机子进程，command/args/env）、`sse`/`streamable-http`
（远程 MCP server，url/headers，headers 通常用于鉴权）。`_open_transport()` 按
`McpServerConfig.transport` 分派，三者产出的 `(read, write)` 流对象形状一致，`connect()`
之后的握手/工具发现逻辑不需要感知具体是哪种 transport。

连接自愈：官方 SDK 不会主动通知 transport 意外关闭（`ClientSession` 的后台接收循环只是在
流耗尽时静默结束，见 `mcp.shared.session.BaseSession._receive_loop`），所以看门狗任务用
`send_ping()` 周期性主动探活，而不是等一个永远不会来的被动回调。探活失败后触发自愈重连——
单飞（`_reconnect_lock`，同一时刻只有一次重连尝试在跑）+ 冷却 + 指数退避（复用
`brain/backoff.py`，不重新发明一套），达到 `max_reconnect_retries` 上限后停止自动重试，
状态定格在 `ConnectionState.ERROR`，可通过 `status()` 查询，等待上层显式调用 `reconnect()`。
看门狗任务的启停完全在本类内部完成（`connect()` 成功后启动、`disconnect()`/`reconnect()`
显式取消）——自愈内部直接调用 `_do_connect()` 而不是 `connect()`/`reconnect()`：前者的早退
guard 只在非 CONNECTED/CONNECTING 状态才会真正发起连接，后者会先取消看门狗任务，而自愈本身
就跑在这个看门狗任务里，取消不了自己所在的任务。

不监听服务端 `notifications/tools/list_changed` 主动通知来热更新工具列表——需要时可以再补，
热更新与手动 `reconnect()` 都会重新跑 `tools/list`，效果等价。

transport 生命周期的任务归属：`stdio_client`/`ClientSession` 内部用 `anyio.create_task_group()`
管理取消范围，而 anyio 的 cancel scope 强制要求"进入"和"退出"必须发生在完全相同的
`asyncio.Task` 上（不匹配就抛 `RuntimeError: Attempted to exit cancel scope in a different
task than it was entered in`）。看门狗在独立后台任务里自愈重连，之后又可能被另一个任务
`disconnect()`/`reconnect()`——如果直接把 `AsyncExitStack` 存成跨任务共享的
`self._exit_stack` 再从别的任务 `aclose()`，就会踩中这条约束。所以 `_do_connect()` 不这么
做：它把"打开 transport→初始化→挂起等关闭信号→关闭 transport"整段 `async with` 都放进
`_connection_body()`，跑在专属的 `self._conn_task` 里，从头到尾不换任务；外部想关闭时只需
`self._closing_event.set()` 再 `await self._conn_task`——这两步对跨任务都是安全的（`Event`
不区分任务，`await` 一个别的任务也不受 anyio 这条约束限制），真正的 `__aexit__` 始终由
`_connection_body` 自己在自己身上执行。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
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

from miku_on_desk.brain.backoff import (
    DEFAULT_BASE_DELAY_S,
    DEFAULT_MAX_DELAY_S,
    DEFAULT_MAX_RETRIES,
    backoff_delay,
)
from miku_on_desk.config.settings import McpServerConfig, McpTransport

logger = logging.getLogger(__name__)

_WATCHDOG_INTERVAL_S = 30.0


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
    reconnect_attempts: int = 0


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

    def __init__(
        self,
        config: McpServerConfig,
        *,
        watchdog_interval_s: float = _WATCHDOG_INTERVAL_S,
        max_reconnect_retries: int = DEFAULT_MAX_RETRIES,
        base_delay_s: float = DEFAULT_BASE_DELAY_S,
        max_delay_s: float = DEFAULT_MAX_DELAY_S,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._config = config
        self._state = ConnectionState.DISCONNECTED
        self._error_message: str | None = None
        self._session: ClientSession | None = None
        self._conn_task: asyncio.Task[None] | None = None
        self._closing_event: asyncio.Event | None = None
        self._tools: list[types.Tool] = []
        self._watchdog_interval_s = watchdog_interval_s
        self._max_reconnect_retries = max_reconnect_retries
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s
        self._sleep = sleep
        self._watchdog_task: asyncio.Task[None] | None = None
        self._reconnect_lock = asyncio.Lock()
        self._reconnect_attempts = 0
        self._generation = 0

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def trusted(self) -> bool:
        return self._config.trusted

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
            reconnect_attempts=self._reconnect_attempts,
        )

    async def connect(self) -> None:
        if self._state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
            return
        self._state = ConnectionState.CONNECTING
        await self._do_connect()

    async def _do_connect(self) -> None:
        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        closing = asyncio.Event()
        self._closing_event = closing
        self._conn_task = asyncio.create_task(self._connection_body(ready, closing))
        try:
            await ready
        except Exception as exc:
            self._state = ConnectionState.ERROR
            self._error_message = str(exc)
            await self._cleanup()
            raise
        self._state = ConnectionState.CONNECTED
        self._error_message = None
        self._reconnect_attempts = 0
        self._generation += 1
        self._start_watchdog()
        logger.info('MCP server "%s" 已连接，发现 %d 个工具', self._config.name, len(self._tools))

    async def _connection_body(self, ready: asyncio.Future[None], closing: asyncio.Event) -> None:
        """在专属任务里跑完整个 transport 生命周期，见模块文档"transport 生命周期的任务归属"。"""
        try:
            async with AsyncExitStack() as stack:
                read, write = await self._open_transport(stack)
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._session = session
                await self._discover_tools()
                ready.set_result(None)
                await closing.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                logger.warning(
                    'MCP server "%s" 关闭连接时出错', self._config.name, exc_info=True
                )
        finally:
            self._session = None

    async def disconnect(self) -> None:
        await self._stop_watchdog()
        await self._cleanup()
        self._state = ConnectionState.DISCONNECTED
        self._error_message = None

    async def reconnect(self) -> None:
        await self._stop_watchdog()
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
        task = self._conn_task
        closing = self._closing_event
        self._conn_task = None
        self._closing_event = None
        if task is None:
            return
        assert closing is not None
        closing.set()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def _start_watchdog(self) -> None:
        if self._watchdog_task is not None and not self._watchdog_task.done():
            return
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def _stop_watchdog(self) -> None:
        task = self._watchdog_task
        if task is None:
            return
        self._watchdog_task = None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _watchdog_loop(self) -> None:
        while True:
            await self._sleep(self._watchdog_interval_s)
            if self._state != ConnectionState.CONNECTED:
                continue
            if await self._check_alive():
                continue
            logger.warning('MCP server "%s" 探活失败，触发自愈重连', self._config.name)
            await self._self_heal()

    async def _check_alive(self) -> bool:
        assert self._session is not None
        try:
            await self._session.send_ping()
        except Exception:
            return False
        return True

    async def _self_heal(self) -> None:
        """单飞 + 冷却 + 指数退避重连，达到重试上限后放弃并定格在 ERROR 状态。

        直接调用 `_do_connect()` 而不是 `connect()`/`reconnect()`：`connect()` 的早退
        guard 只在状态是 DISCONNECTED/ERROR 时才会真正发起连接，这里必须先把状态从
        CONNECTED 挪到 CONNECTING 才能绕开这道 guard；`reconnect()` 会先 `_stop_watchdog()`，
        而这个方法本身就跑在被取消的那个看门狗任务里，自己取消自己会导致重连中途夭折。

        `generation` 在获取锁之前先记下来，获取锁之后重新核对：并发触发时，后到者等锁期间，
        世界可能已经被先到者的自愈重连改变过（状态又回到了 CONNECTED）——只看 `state ==
        CONNECTED` 分不清"从未坏过"和"刚被别人治好"，必须靠代际计数确认"我发现故障时的那一代
        连接还是不是现在这一代"，不是才回收手，避免同一次故障被治两遍。
        """
        generation = self._generation
        async with self._reconnect_lock:
            if self._generation != generation or self._state != ConnectionState.CONNECTED:
                return
            self._state = ConnectionState.CONNECTING
            await self._cleanup()
            attempt = 0
            while True:
                try:
                    await self._do_connect()
                    return
                except Exception:
                    if attempt >= self._max_reconnect_retries:
                        logger.error(
                            'MCP server "%s" 自愈重连耗尽 %d 次重试，放弃，等待手动 reconnect()',
                            self._config.name,
                            self._max_reconnect_retries,
                        )
                        return
                    delay = backoff_delay(
                        attempt, base_delay_s=self._base_delay_s, max_delay_s=self._max_delay_s
                    )
                    logger.warning(
                        'MCP server "%s" 自愈重连第 %d 次失败，%.2f 秒后重试',
                        self._config.name,
                        attempt + 1,
                        delay,
                        exc_info=True,
                    )
                    await self._sleep(delay)
                    attempt += 1
                    self._reconnect_attempts = attempt
