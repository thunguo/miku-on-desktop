"""MCPHost 的工具桥接/生命周期回归测试，同样对着真实的 `_fixture_server.py` 子进程跑。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from miku_on_desk.brain.mcp.client import ConnectionState
from miku_on_desk.brain.mcp.host import MCPHost
from miku_on_desk.brain.providers.base import ToolUseBlock
from miku_on_desk.brain.tools.path_sandbox import PathSandbox
from miku_on_desk.brain.tools.policy import Decision, PolicyEngine
from miku_on_desk.brain.tools.read_tracker import ReadTracker
from miku_on_desk.brain.tools.registry import ToolRegistry
from miku_on_desk.config.settings import McpServerConfig

_FIXTURE_SERVER = Path(__file__).parent / "_fixture_server.py"


def _fixture_config(name: str = "fixture") -> McpServerConfig:
    return McpServerConfig(name=name, command=sys.executable, args=[str(_FIXTURE_SERVER)])


@pytest.fixture
def registry(tmp_path: Path) -> ToolRegistry:
    sandbox = PathSandbox(cwd=tmp_path, output_dir=tmp_path, data_dir=tmp_path)
    policy = PolicyEngine(
        trusted_mode=True,
        allowed_tools=frozenset(),
        denied_tools=frozenset(),
        default_decision=Decision.ALLOW,
        path_sandbox=sandbox,
        read_tracker=ReadTracker(),
    )
    return ToolRegistry(policy, ReadTracker())


@pytest.fixture
async def host(registry: ToolRegistry) -> MCPHost:
    h = MCPHost(registry)
    yield h
    await h.shutdown()


async def test_connect_server_bridges_tools_with_namespaced_names(
    host: MCPHost, registry: ToolRegistry
) -> None:
    await host.connect_server(_fixture_config())

    names = {d.name for d in registry.definitions()}
    assert names == {"mcp_fixture_echo", "mcp_fixture_fail"}


async def test_bridged_tool_description_carries_server_prefix(
    host: MCPHost, registry: ToolRegistry
) -> None:
    await host.connect_server(_fixture_config())

    echo_def = next(d for d in registry.definitions() if d.name == "mcp_fixture_echo")
    assert echo_def.description.startswith("[MCP:fixture]")


async def test_bridged_tool_executes_through_to_the_mcp_server(
    host: MCPHost, registry: ToolRegistry
) -> None:
    await host.connect_server(_fixture_config())

    result = await registry.execute(
        ToolUseBlock(id="call1", name="mcp_fixture_echo", input={"text": "路由器"}),
        session_id="s1",
    )

    assert result.is_error is False
    assert "路由器" in result.content


async def test_bridged_tool_reports_error_result_on_server_failure(
    host: MCPHost, registry: ToolRegistry
) -> None:
    await host.connect_server(_fixture_config())

    result = await registry.execute(
        ToolUseBlock(id="call1", name="mcp_fixture_fail", input={"reason": "坏了"}),
        session_id="s1",
    )

    assert result.is_error is True


async def test_disconnect_server_unregisters_its_tools(
    host: MCPHost, registry: ToolRegistry
) -> None:
    await host.connect_server(_fixture_config())

    await host.disconnect_server("fixture")

    assert registry.definitions() == []


async def test_disconnect_server_does_not_affect_other_servers(
    host: MCPHost, registry: ToolRegistry
) -> None:
    await host.connect_server(_fixture_config("one"))
    await host.connect_server(_fixture_config("two"))

    await host.disconnect_server("one")

    names = {d.name for d in registry.definitions()}
    assert names == {"mcp_two_echo", "mcp_two_fail"}


async def test_initialize_connects_all_enabled_servers_and_skips_disabled(
    host: MCPHost, registry: ToolRegistry
) -> None:
    await host.initialize(
        [
            _fixture_config("one"),
            _fixture_config("two"),
            McpServerConfig(
                name="disabled", command=sys.executable, args=[str(_FIXTURE_SERVER)], enabled=False
            ),
        ]
    )

    names = {d.name for d in registry.definitions()}
    assert names == {"mcp_one_echo", "mcp_one_fail", "mcp_two_echo", "mcp_two_fail"}


async def test_initialize_isolates_a_single_failing_server(
    host: MCPHost, registry: ToolRegistry
) -> None:
    await host.initialize(
        [
            _fixture_config("good"),
            McpServerConfig(name="bad", command="this-binary-does-not-exist-anywhere"),
        ]
    )

    names = {d.name for d in registry.definitions()}
    assert names == {"mcp_good_echo", "mcp_good_fail"}
    assert host.get_server_status("bad") is not None
    assert host.get_server_status("bad").state == ConnectionState.ERROR


async def test_list_servers_reports_status_for_every_connection(
    host: MCPHost, registry: ToolRegistry
) -> None:
    await host.connect_server(_fixture_config("one"))
    await host.connect_server(_fixture_config("two"))

    statuses = {status.name: status for status in host.list_servers()}
    assert statuses.keys() == {"one", "two"}
    assert all(status.state == ConnectionState.CONNECTED for status in statuses.values())


async def test_reconnect_server_rebridges_tools(host: MCPHost, registry: ToolRegistry) -> None:
    await host.connect_server(_fixture_config())

    await host.reconnect_server("fixture")

    names = {d.name for d in registry.definitions()}
    assert names == {"mcp_fixture_echo", "mcp_fixture_fail"}


async def test_reconnect_server_raises_key_error_when_not_connected(host: MCPHost) -> None:
    with pytest.raises(KeyError):
        await host.reconnect_server("does-not-exist")


async def test_shutdown_disconnects_all_servers_and_clears_tools(
    host: MCPHost, registry: ToolRegistry
) -> None:
    await host.connect_server(_fixture_config("one"))
    await host.connect_server(_fixture_config("two"))

    await host.shutdown()

    assert registry.definitions() == []
    assert host.list_servers() == []
