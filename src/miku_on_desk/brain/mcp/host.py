"""MCP host：按配置批量连接外部 MCP server，把它们的工具并入统一 ``ToolRegistry``。

`initialize()` 用 ``asyncio.gather(..., return_exceptions=True)``，单个 server 连接失败不
影响其余 server；工具桥接的命名规则：工具名清洗为 ``[a-zA-Z0-9_-]``（Anthropic 工具名约束
``^[a-zA-Z0-9_-]{1,128}$``），命名空间前缀 ``mcp_<server>_<tool>``，描述前缀 ``[MCP:<server>]``。

暴露 server 配置前不需要脱敏 ``Authorization`` header：`McpServerConfig` 虽然有 ``headers``
字段，但 `MCPServerStatus`（`client.py`）只暴露 name/state/tool_count/error_message 四个
字段，从不包含 headers/url；日志里也从不整个序列化 config（`connect()` 的 `logger.info`
只插值 name 和工具数量）。headers 从不出现在任何暴露面上，因此不需要额外脱敏代码。

外部 MCP server 的工具对我们来说是黑盒——不像 `builtin/` 工具那样能在注册时手写一个精确的
`ToolPolicySpec`，所以 `_infer_policy_spec()` 按 `inputSchema` 里的参数名做启发式推断：
命中 `path`/`file_path`/`filepath` 就接入路径沙箱，命中 `command`/`cmd` 就接入危险命令
检测，且默认 `requires_confirmation=True`——比 `ToolPolicySpec()` 的默认值更保守，呼应
`policy.py` 里"结构性边界优先于信任层"的公理：MCP 桥接工具不例外，不会因为"来自 MCP"就
默认更被信任。`McpServerConfig.trusted` 只豁免这里的 `requires_confirmation`，路径沙箱/
先读后改这两条结构性检查始终生效，由 `PolicyEngine.evaluate()` 统一把关。
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from typing import Any

from mcp import types

from miku_on_desk.brain.mcp.client import MCPServerConnection, MCPServerStatus, MCPToolError
from miku_on_desk.brain.providers.base import ToolDefinition
from miku_on_desk.brain.tools.policy import ToolPolicySpec
from miku_on_desk.brain.tools.registry import (
    ToolExecutionError,
    ToolHandler,
    ToolRegistration,
    ToolRegistry,
)
from miku_on_desk.config.settings import McpServerConfig

logger = logging.getLogger(__name__)

_EMPTY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}
_PATH_PARAM_NAMES = frozenset({"path", "file_path", "filepath"})
_COMMAND_PARAM_NAMES = frozenset({"command", "cmd"})


def _infer_policy_spec(tool: types.Tool, *, trusted: bool) -> ToolPolicySpec:
    """按 `tool.inputSchema` 里的参数名推断这个未知工具要接入哪些结构性防御闸门。"""
    properties: dict[str, Any] = (tool.inputSchema or {}).get("properties", {})
    path_arg = next((name for name in properties if name in _PATH_PARAM_NAMES), None)
    command_arg = next((name for name in properties if name in _COMMAND_PARAM_NAMES), None)
    return ToolPolicySpec(
        requires_confirmation=not trusted,
        confirm_reason=f'MCP 工具 "{tool.name}" 需要用户确认后才能执行。',
        command_arg=command_arg,
        path_arg=path_arg,
    )


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


class MCPHost:
    """管理所有 MCP server 连接的生命周期，并维护它们在 ``ToolRegistry`` 里的工具镜像。"""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._connections: dict[str, MCPServerConnection] = {}

    async def initialize(self, configs: Sequence[McpServerConfig]) -> None:
        enabled = [config for config in configs if config.enabled]
        if not enabled:
            logger.info("未配置任何 MCP server")
            return
        results = await asyncio.gather(
            *(self.connect_server(config) for config in enabled), return_exceptions=True
        )
        failed = sum(1 for result in results if isinstance(result, BaseException))
        logger.info(
            "MCP server 初始化完成：%d 个成功，%d 个失败", len(enabled) - failed, failed
        )

    async def connect_server(self, config: McpServerConfig) -> None:
        if config.name in self._connections:
            await self.disconnect_server(config.name)
        connection = MCPServerConnection(config)
        try:
            await connection.connect()
        except Exception:
            logger.warning('MCP server "%s" 连接失败', config.name, exc_info=True)
            raise
        finally:
            self._connections[config.name] = connection
        self._register_tools(connection)

    async def disconnect_server(self, name: str) -> None:
        connection = self._connections.pop(name, None)
        if connection is None:
            return
        self._unregister_tools(connection)
        await connection.disconnect()

    async def reconnect_server(self, name: str) -> None:
        connection = self._connections.get(name)
        if connection is None:
            raise KeyError(f'MCP server "{name}" 未连接。')
        self._unregister_tools(connection)
        await connection.reconnect()
        self._register_tools(connection)

    async def shutdown(self) -> None:
        # 官方 SDK 的 stdio_client 内部用 anyio.create_task_group()，同一个 task 里的多个
        # cancel scope 必须严格按"后开先关"关闭；否则会抛
        # RuntimeError: Attempted to exit a cancel scope that isn't the current task's
        # current cancel scope（上游已知问题，见 modelcontextprotocol/python-sdk#577）。
        for name in reversed(list(self._connections)):
            await self.disconnect_server(name)

    def list_servers(self) -> list[MCPServerStatus]:
        return [connection.status() for connection in self._connections.values()]

    def get_server_status(self, name: str) -> MCPServerStatus | None:
        connection = self._connections.get(name)
        return connection.status() if connection is not None else None

    def _register_tools(self, connection: MCPServerConnection) -> None:
        safe_server = _sanitize(connection.name)
        for tool in connection.tools:
            full_name = f"mcp_{safe_server}_{_sanitize(tool.name)}"
            self._registry.register(
                ToolRegistration(
                    definition=ToolDefinition(
                        name=full_name,
                        description=f"[MCP:{connection.name}] {tool.description or tool.name}",
                        input_schema=tool.inputSchema or _EMPTY_OBJECT_SCHEMA,
                    ),
                    handler=_make_handler(connection, tool.name),
                    policy_spec=_infer_policy_spec(tool, trusted=connection.trusted),
                )
            )

    def _unregister_tools(self, connection: MCPServerConnection) -> None:
        prefix = f"mcp_{_sanitize(connection.name)}_"
        for definition in self._registry.definitions():
            if definition.name.startswith(prefix):
                self._registry.unregister(definition.name)


def _make_handler(connection: MCPServerConnection, original_tool_name: str) -> ToolHandler:
    async def handler(tool_input: dict[str, Any]) -> str:
        try:
            return await connection.call_tool(original_tool_name, tool_input)
        except MCPToolError as exc:
            raise ToolExecutionError(str(exc)) from exc

    return handler
