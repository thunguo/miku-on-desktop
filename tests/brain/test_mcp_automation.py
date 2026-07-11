"""``build_automation_tool_use`` 的拼接规则需要与 ``brain/mcp/host.py`` 里真实注册工具时
使用的 ``full_name`` 格式完全一致——否则自动化触发的工具名永远查不到 ``ToolRegistry`` 里
真实注册的条目。这里独立复制同一份 ``_sanitize`` 正则规则来做对照，不导入 ``host.py`` 的
私有符号。
"""

from __future__ import annotations

import re

from miku_on_desk.brain.mcp_automation import build_automation_tool_use
from miku_on_desk.config.settings import McpAutomationConfig


def _host_full_name(server_name: str, tool_name: str) -> str:
    sanitize = re.compile(r"[^a-zA-Z0-9_-]")
    safe_server = sanitize.sub("_", server_name)
    return f"mcp_{safe_server}_{sanitize.sub('_', tool_name)}"


def test_build_automation_tool_use_matches_host_registration_naming() -> None:
    config = McpAutomationConfig(server_name="Spotify Player", tool_name="play/track")
    tool_use = build_automation_tool_use(config)
    assert tool_use.name == _host_full_name("Spotify Player", "play/track")


def test_build_automation_tool_use_passes_tool_input_unchanged() -> None:
    tool_input = {"uri": "spotify:track:123", "volume": 0.5}
    config = McpAutomationConfig(server_name="spotify", tool_name="play", tool_input=tool_input)
    tool_use = build_automation_tool_use(config)
    assert tool_use.input == tool_input


def test_build_automation_tool_use_generates_unique_id_each_call() -> None:
    config = McpAutomationConfig(server_name="spotify", tool_name="play")
    first = build_automation_tool_use(config)
    second = build_automation_tool_use(config)
    assert first.id != second.id
    assert first.id.startswith("mcp-automation-")
    assert second.id.startswith("mcp-automation-")
