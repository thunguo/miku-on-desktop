"""ACP（Agent Client Protocol）单次任务委派：把一句任务描述转发给本机已安装的外部编码
agent（如 Claude Code、Codex 的 ACP 模式），拿回它跑完一整个 turn 后的最终文本。

和 `agents/spawn.py` 的 `spawn_agents` 不同：那边派发给的是 Brain 内部复用同一套
Provider/工具的子循环；这里委派的是完全独立的外部进程，走 stdio JSON-RPC，可能装了完全不同
的工具链（比如 Claude Code 自己的文件编辑/终端工具）。协议层直接用官方 `agent-client-protocol`
SDK（导入名 `acp`），不手写 JSON-RPC framing——SDK 自带的 `spawn_agent_process` 已经处理了子进程
生命周期（含 stdin EOF→drain→close→wait_closed，超时后 terminate 再 kill 的防御式关闭），
不需要重新发明。

`ClientCapabilities` 全部留空（`fs.read_text_file`/`write_text_file`/`terminal` 均为默认的
`False`），这不是遗漏而是有意选择：ACP 最初是为浏览器/编辑器里"文件可能只存在于未保存的缓冲区"
这种场景设计的委派方——但这里的外部 agent 本身就是本机真实进程，对磁盘有和我们平级的直接访问
权限，通过我们中转文件 I/O 只会多一层没有必要的间接层。因此 `_AcpSessionClient` 里
`read_text_file`/`write_text_file`/`create_terminal` 等方法几乎不会被调用——按 ACP 规范，
遵从协议的 agent 只应在对应能力被声明为 true 时才发起这些请求；这里仍然给出正确实现（文件 I/O）
或明确的 `method_not_found`（终端相关），而不是留空占位，一是防御不完全遵从协议的 agent，
二是 `_AcpSessionClient` 要作为具体实例传给 `spawn_agent_process(to_client: ... | Client, ...)`，
必须结构性满足 `acp.interfaces.Client` 这个 Protocol 的全部方法才能通过 mypy strict 检查。

授权确认：`request_permission` 里遇到 `allow_once`/`allow_always` 选项即自动选择——这个委派
本身就是主循环里一次已经通过七层权限闸门 ASK/DENY 判断之后才会被调用的工具，外部 agent 自己
再问一遍"要不要执行这个操作"没有对应的 UI 通道可以转发（Miku 只有一个确认气泡，没有能力代表
两个独立 agent 分别确认），所以这里对齐 `agents/spawn.py` 的 `_auto_approve`：委派出去之后，
外部 agent 内部的确认全部自动放行，不做二次拦截。

尽管上一段解释了为什么遵从协议的 agent 基本不会调用 `read_text_file`/`write_text_file`，
`_AcpSessionClient` 仍然给这两个方法接入可选的 `path_sandbox: PathSandbox`——防御"不完全遵从
协议但仍尝试发起文件 I/O 请求"的外部 agent（`ClientCapabilities` 只是声明,不是强制约束，
SDK 不会替我们拦下违反声明的请求）。拒绝时用 `RequestError.invalid_params`：ACP 错误码里没有
专门对应"路径不在允许范围"的语义，这是几个通用错误码里最贴切的一个。

握手重试：`_handshake()`（进程启动 + `initialize` + `new_session`）失败会换一个全新子进程、
按 `brain/backoff.py` 的指数退避重试有限次数——这个阶段任何内容都还没有流出给 UI，重试不会
产生重复/错乱的可见输出。一旦握手成功进入 `conn.prompt()`，就不再属于握手范围，遵循
`providers/retry.py` 定下的规则：内容已经开始流出后不重试。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, Agent, RequestError, spawn_agent_process, text_block
from acp.schema import (
    AgentMessageChunk,
    AllowedOutcome,
    CreateTerminalResponse,
    DeniedOutcome,
    Implementation,
    KillTerminalResponse,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    TerminalOutputResponse,
    ToolCallUpdate,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)

from miku_on_desk.brain.backoff import (
    DEFAULT_BASE_DELAY_S,
    DEFAULT_MAX_DELAY_S,
    DEFAULT_MAX_RETRIES,
    backoff_delay,
)
from miku_on_desk.brain.tools.path_sandbox import PathSandbox

logger = logging.getLogger(__name__)

_CLIENT_NAME = "miku-on-desk"
_CLIENT_VERSION = "0.1.0"
_DEFAULT_TIMEOUT_S = 900.0


@dataclass(frozen=True)
class AcpTurnResult:
    success: bool
    content: str
    error: str | None
    stop_reason: str | None


class _AcpSessionClient:
    """满足 `acp.interfaces.Client` Protocol 的最小实现，只服务单次 `run_acp_task` 调用。"""

    def __init__(
        self,
        on_chunk: Callable[[str], None] | None = None,
        *,
        path_sandbox: PathSandbox | None = None,
    ) -> None:
        self._chunks: list[str] = []
        self._on_chunk = on_chunk
        self._path_sandbox = path_sandbox

    @property
    def text(self) -> str:
        return "".join(self._chunks)

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        if isinstance(update, AgentMessageChunk) and update.content.type == "text":
            chunk_text = update.content.text
            self._chunks.append(chunk_text)
            if self._on_chunk is not None:
                self._on_chunk(chunk_text)

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        for option in options:
            if option.kind in ("allow_once", "allow_always"):
                return RequestPermissionResponse(
                    outcome=AllowedOutcome(outcome="selected", option_id=option.option_id)
                )
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **kwargs: Any,
    ) -> ReadTextFileResponse:
        self._check_sandbox(path)
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise RequestError.resource_not_found(path) from exc
        lines = text.splitlines(keepends=True)
        if line is not None:
            lines = lines[line:]
        if limit is not None:
            lines = lines[:limit]
        return ReadTextFileResponse(content="".join(lines))

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: Any
    ) -> WriteTextFileResponse | None:
        self._check_sandbox(path)
        Path(path).write_text(content, encoding="utf-8")
        return None

    def _check_sandbox(self, raw_path: str) -> None:
        if self._path_sandbox is None:
            return
        result = self._path_sandbox.check(Path(raw_path))
        if not result.allowed:
            raise RequestError.invalid_params(data={"path": raw_path, "reason": result.reason})

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[Any] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> TerminalOutputResponse:
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None:
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalResponse | None:
        raise RequestError.method_not_found("terminal/kill")

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(f"_{method}")

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None

    def on_connect(self, conn: Agent) -> None:
        return None


async def _handshake(
    *,
    client: _AcpSessionClient,
    executable: str,
    args: Sequence[str],
    cwd: str,
    env: Mapping[str, str] | None,
    stack: AsyncExitStack,
) -> tuple[Agent, str]:
    """进入 `spawn_agent_process` 上下文并完成 `initialize` + `new_session`。

    整个子进程生命周期都挂在 `stack` 上——握手失败时调用方 `await stack.aclose()` 即可
    干净地杀掉这次进程，换一个新的 `AsyncExitStack` 重新调用本函数重试。
    """
    conn, _process = await stack.enter_async_context(
        spawn_agent_process(client, executable, *args, cwd=cwd, env=env)
    )
    await conn.initialize(
        protocol_version=PROTOCOL_VERSION,
        client_info=Implementation(name=_CLIENT_NAME, version=_CLIENT_VERSION),
    )
    session = await conn.new_session(cwd=cwd, mcp_servers=[])
    return conn, session.session_id


async def _run_turn(
    *,
    executable: str,
    args: Sequence[str],
    cwd: str,
    task: str,
    env: Mapping[str, str] | None,
    on_chunk: Callable[[str], None] | None,
    path_sandbox: PathSandbox | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
    max_delay_s: float = DEFAULT_MAX_DELAY_S,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> AcpTurnResult:
    client = _AcpSessionClient(on_chunk=on_chunk, path_sandbox=path_sandbox)

    attempt = 0
    while True:
        async with AsyncExitStack() as stack:
            try:
                conn, session_id = await _handshake(
                    client=client, executable=executable, args=args, cwd=cwd, env=env, stack=stack
                )
            except Exception:
                if attempt >= max_retries:
                    raise
                delay = backoff_delay(attempt, base_delay_s=base_delay_s, max_delay_s=max_delay_s)
                logger.warning(
                    'ACP 握手第 %d 次失败，%.2f 秒后换一个新进程重试："%s"',
                    attempt + 1,
                    delay,
                    executable,
                    exc_info=True,
                )
                await sleep(delay)
                attempt += 1
                continue

            response = await conn.prompt(prompt=[text_block(task)], session_id=session_id)

        success = response.stop_reason == "end_turn"
        error = None if success else f"ACP 任务未正常结束（stop_reason={response.stop_reason}）"
        return AcpTurnResult(
            success=success, content=client.text, error=error, stop_reason=response.stop_reason
        )


async def run_acp_task(
    *,
    executable: str,
    args: Sequence[str] = (),
    cwd: str,
    task: str,
    env: Mapping[str, str] | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    on_chunk: Callable[[str], None] | None = None,
    path_sandbox: PathSandbox | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
    max_delay_s: float = DEFAULT_MAX_DELAY_S,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> AcpTurnResult:
    """启动 `executable`（ACP 模式的外部 agent 进程），委派一次任务，返回其最终回复文本。

    `on_chunk` 在每次收到一个消息分片时同步调用一次，用于把长时间委派期间的中间输出实时
    转发给 UI；不影响 `client.text` 这个最终兜底缓冲区，两者并行写入。

    `path_sandbox` 为 None 时（默认）不做任何拦截，与接入前行为一致；传入后为
    `_AcpSessionClient.read_text_file`/`write_text_file` 接入结构性边界，见模块文档。

    `max_retries`/`base_delay_s`/`max_delay_s`/`sleep` 只影响握手阶段（进程启动到
    `new_session` 之间）的重试；一旦进入 `prompt()`，失败就直接返回，不重试。
    """
    try:
        return await asyncio.wait_for(
            _run_turn(
                executable=executable,
                args=args,
                cwd=cwd,
                task=task,
                env=env,
                on_chunk=on_chunk,
                path_sandbox=path_sandbox,
                max_retries=max_retries,
                base_delay_s=base_delay_s,
                max_delay_s=max_delay_s,
                sleep=sleep,
            ),
            timeout=timeout_s,
        )
    except TimeoutError:
        return AcpTurnResult(
            success=False, content="", error=f"ACP 任务超时（{timeout_s:.0f} 秒）", stop_reason=None
        )
    except Exception as exc:
        logger.exception('ACP 任务委派给 "%s" 时抛出未预期异常', executable)
        return AcpTurnResult(success=False, content="", error=str(exc), stop_reason=None)
