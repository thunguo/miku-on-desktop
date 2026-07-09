"""exec_command：在本机 shell 里执行命令，是 read_file/write_file 路径沙箱之外的既定逃生舱
（``path_sandbox.py`` 拒绝文案里早就把它写成"沙箱外改用 exec_command"的官方路径，这里只是把
预留的名字实现出来）——因此这个工具故意不做 path_arg 校验，风险完全靠 requires_confirmation +
输出截断 + 超时来兜底。
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, ValidationError

from miku_on_desk.brain.providers.base import ToolDefinition
from miku_on_desk.brain.tools.policy import ToolPolicySpec
from miku_on_desk.brain.tools.registry import (
    ToolExecutionError,
    ToolHandler,
    ToolRegistration,
    ToolRegistry,
)

_TIMEOUT_S = 30.0
_MAX_OUTPUT_CHARS = 20_000


class ExecCommandInput(BaseModel):
    command: str


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    omitted = len(text) - _MAX_OUTPUT_CHARS
    return text[:_MAX_OUTPUT_CHARS] + f"\n\n...[输出已截断，省略了后面 {omitted} 字符]"


def _make_exec_command_handler() -> ToolHandler:
    async def handler(tool_input: dict[str, Any]) -> str:
        try:
            parsed = ExecCommandInput.model_validate(tool_input)
        except ValidationError as exc:
            raise ToolExecutionError(f"参数不合法：{exc}") from exc

        try:
            proc = await asyncio.create_subprocess_shell(
                parsed.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            raise ToolExecutionError(f"命令启动失败：{exc}") from exc

        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT_S)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise ToolExecutionError(f"命令执行超过 {_TIMEOUT_S:.0f} 秒，已被终止。") from None

        output = _truncate(stdout.decode("utf-8", errors="replace"))
        return f"exit_code={proc.returncode}\n{output}"

    return handler


def register_exec_command_tool(registry: ToolRegistry) -> None:
    registry.register(
        ToolRegistration(
            definition=ToolDefinition(
                name="exec_command",
                description=(
                    "在本机 shell 里执行一条命令并返回其标准输出/错误（已合并）与退出码。"
                    "不受 read_file/write_file 的路径沙箱限制，可以用来访问沙箱之外的路径"
                    f"（如 cat/ls）。默认 {_TIMEOUT_S:.0f} 秒超时会杀掉进程；输出超过 "
                    f"{_MAX_OUTPUT_CHARS} 字符会被截断。每次执行都需要用户二次确认。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "要执行的 shell 命令"},
                    },
                    "required": ["command"],
                },
            ),
            handler=_make_exec_command_handler(),
            policy_spec=ToolPolicySpec(
                command_arg="command",
                requires_confirmation=True,
                confirm_reason="即将在本机执行一条 shell 命令，是否允许？",
            ),
        )
    )
