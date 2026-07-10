"""基于 hook 事件自动触发一次预配置的 MCP 工具调用（"一键场景模板"的运行时落点）。

工具名拼接规则独立复制自 ``brain/mcp/host.py::_sanitize``/``_register_tools``，不跨模块
导入其私有符号——两处各自维护同一份 ``re.sub(r"[^a-zA-Z0-9_-]", "_", name)`` 规则,只要
``server_name``/``tool_name`` 与真实注册时使用的 MCP server 名/工具名一致,拼出来的
``mcp_<server>_<tool>`` 就必然命中 ``ToolRegistry`` 里真实注册的工具;若拼错或目标 server
未连接,``ToolRegistry.evaluate()`` 会返回 DENY,这是已有的干净失败路径,这里不需要重复校验。
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from miku_on_desk.brain.providers.base import ToolUseBlock
from miku_on_desk.config.settings import McpAutomationConfig


@dataclass(frozen=True)
class McpAutomationTrigger:
    hook_event_name: str


def _sanitize_mcp_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def build_automation_tool_use(config: McpAutomationConfig) -> ToolUseBlock:
    server = _sanitize_mcp_name(config.server_name)
    tool = _sanitize_mcp_name(config.tool_name)
    return ToolUseBlock(
        id=f"mcp-automation-{uuid.uuid4().hex}",
        name=f"mcp_{server}_{tool}",
        input=config.tool_input,
    )
