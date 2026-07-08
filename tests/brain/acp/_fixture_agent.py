"""供 ACP 集成测试使用的最小外部 agent，用官方 `agent-client-protocol` SDK 的 `run_agent`
在真实 stdio 子进程里跑起来——不是手写假连接对象，验证的是 `run_acp_task`/`_AcpSessionClient`
真的能对上 SDK 的线上协议行为。

按收到的 prompt 文本分发到不同的测试场景（模仿 `_fixture_server.py` 用不同工具名区分场景的
做法，但 ACP 只有一个 `prompt` 入口，所以改用文本前缀分发）：
- `echo:<text>`：回一条包含 `<text>` 的消息分片，end_turn。
- `echo_multi:<a>|<b>|<c>`：按 `|` 分隔依次回多条消息分片，end_turn，用于验证流式分片
  按顺序逐条到达（而非等最终结果才拿到）。
- `refuse`：不回任何消息分片，直接以 `refusal` 结束。
- `sleep:<seconds>`：先睡够 `<seconds>` 秒再 end_turn，用于验证超时路径。
- `request_permission`：主动发一次权限请求，把 Client 选中的 option_id 回显在消息分片里，
  用于验证 `_AcpSessionClient.request_permission` 的自动批准逻辑。
- `write_file:<path>|<content>`/`read_file:<path>`：主动调用 `self._conn.write_text_file`/
  `read_text_file`——`ClientCapabilities` 全都声明为 False，遵从协议的 agent 不会发起这两个
  请求，这里故意无视声明主动发起，模拟"不完全遵从协议的外部 agent"，用于验证
  `_AcpSessionClient` 的 `path_sandbox` 拦截逻辑真的会通过协议往返生效（而不是被信任跳过）。

`FIXTURE_FAIL_COUNT_FILE` 环境变量（进程启动时读取，不是 prompt 分发）：文件里存一个剩余
失败次数；非零则本次进程直接 `sys.exit(1)`（模拟握手阶段就崩溃）并把计数减一后写回，用于
测试 `run_acp_task` 的握手重试——因为每次重试都是全新子进程，失败计数必须落在进程外的文件里
才能在多次重试之间累计。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from acp import RequestError, run_agent
from acp.helpers import text_block, update_agent_message
from acp.schema import (
    Implementation,
    InitializeResponse,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
    ToolCallUpdate,
)


class FixtureAgent:
    def __init__(self) -> None:
        self._conn: Any = None

    def on_connect(self, conn: Any) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Any = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_info=Implementation(name="fixture-agent", version="0.1.0"),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: Any = None,
        mcp_servers: Any = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        return NewSessionResponse(session_id="fixture-session")

    async def prompt(
        self, prompt: list[Any], session_id: str, message_id: str | None = None, **kwargs: Any
    ) -> PromptResponse:
        text = prompt[0].text if prompt and prompt[0].type == "text" else ""

        if text == "refuse":
            return PromptResponse(stop_reason="refusal")

        if text.startswith("sleep:"):
            await asyncio.sleep(float(text.removeprefix("sleep:")))
            return PromptResponse(stop_reason="end_turn")

        if text == "request_permission":
            option = PermissionOption(kind="allow_once", name="Allow", option_id="allow-option")
            response = await self._conn.request_permission(
                options=[option],
                session_id=session_id,
                tool_call=ToolCallUpdate(tool_call_id="fixture-call"),
            )
            outcome_text = (
                f"approved:{response.outcome.option_id}"
                if response.outcome.outcome == "selected"
                else "denied"
            )
            await self._conn.session_update(
                session_id=session_id, update=update_agent_message(text_block(outcome_text))
            )
            return PromptResponse(stop_reason="end_turn")

        if text.startswith("echo:"):
            await self._conn.session_update(
                session_id=session_id,
                update=update_agent_message(text_block(text.removeprefix("echo:"))),
            )
            return PromptResponse(stop_reason="end_turn")

        if text.startswith("echo_multi:"):
            for part in text.removeprefix("echo_multi:").split("|"):
                await self._conn.session_update(
                    session_id=session_id, update=update_agent_message(text_block(part))
                )
            return PromptResponse(stop_reason="end_turn")

        if text.startswith("write_file:"):
            path, _, content = text.removeprefix("write_file:").partition("|")
            try:
                await self._conn.write_text_file(content=content, path=path, session_id=session_id)
                outcome_text = "write_ok"
            except RequestError as exc:
                outcome_text = f"write_denied:{exc}"
            await self._conn.session_update(
                session_id=session_id, update=update_agent_message(text_block(outcome_text))
            )
            return PromptResponse(stop_reason="end_turn")

        if text.startswith("read_file:"):
            path = text.removeprefix("read_file:")
            try:
                response = await self._conn.read_text_file(path=path, session_id=session_id)
                outcome_text = f"read_ok:{response.content}"
            except RequestError as exc:
                outcome_text = f"read_denied:{exc}"
            await self._conn.session_update(
                session_id=session_id, update=update_agent_message(text_block(outcome_text))
            )
            return PromptResponse(stop_reason="end_turn")

        return PromptResponse(stop_reason="end_turn")


if __name__ == "__main__":
    fail_count_file = os.environ.get("FIXTURE_FAIL_COUNT_FILE")
    if fail_count_file:
        path = Path(fail_count_file)
        remaining = int(path.read_text()) if path.exists() else 0
        if remaining > 0:
            path.write_text(str(remaining - 1))
            sys.exit(1)
    asyncio.run(run_agent(FixtureAgent()))
