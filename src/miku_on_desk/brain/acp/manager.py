"""ACP 外部 agent 的委派入口：把配置好的本机外部 agent（Claude Code、Codex 等 ACP 模式进程）
解析为可执行命令，并注册成主循环可调用的 `acp_delegate` 工具。

和 `mcp/host.py` 的关系：两者都是"把配置列表里的外部能力接入统一工具注册表"，但 MCP host
维护的是长连接（stdio 进程常驻，启动时枚举一次工具），这里则是每次调用都按需起停子进程（见
`client.py` 顶部说明，复用 SDK 自带的防御式关闭），因为 ACP 天然是"委派一次任务、拿到最终结果
就结束"的一次性交互模型，不是可常驻复用的连接。所以这里不需要 `connect`/`disconnect`/
`reconnect` 这类连接生命周期方法，只需要"按名字解析配置"这一件事。

`acp_delegate` 的 `cwd` 参数在 schema 里是必填项，不是遗漏默认值：外部 agent 要操作哪个目录
必须由调用它的模型明确给出，不能凑一个隐式默认——不同任务指向的目录通常完全不同，猜错目录的
代价（在错误的项目里跑一次自主编码任务）远高于多问模型一个字段。为呼应这一点，`cwd` 走
``ToolPolicySpec.path_arg`` 接入路径沙箱校验：我们不代理外部 agent 对沙箱内文件的访问方式
（见 `client.py` 顶部说明），但仍然限制它能拿到的工作目录必须落在允许范围内。

`register_acp_delegate_tool` 的 `path_sandbox` 原样转发给 `run_acp_task`，接入的是
`_AcpSessionClient.read_text_file`/`write_text_file` 那道防线（见 `client.py` 顶部说明），
与上一段的 `cwd` 沙箱校验是两条独立的结构性边界，互不替代。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from miku_on_desk.brain.acp.client import _DEFAULT_TIMEOUT_S, run_acp_task
from miku_on_desk.brain.providers.base import ToolDefinition
from miku_on_desk.brain.tools.path_sandbox import PathSandbox
from miku_on_desk.brain.tools.policy import ToolPolicySpec
from miku_on_desk.brain.tools.registry import (
    ToolExecutionError,
    ToolHandler,
    ToolRegistration,
    ToolRegistry,
)
from miku_on_desk.config.settings import AcpAgentConfig

logger = logging.getLogger(__name__)

_ACP_DELEGATE_TOOL_NAME = "acp_delegate"


class AcpManager:
    """按名字解析已配置且启用的 ACP 外部 agent；不维护任何长连接状态。"""

    def __init__(
        self,
        agents: Sequence[AcpAgentConfig],
        *,
        default_timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._agents = {agent.name: agent for agent in agents}
        self._default_timeout_s = default_timeout_s

    @property
    def default_timeout_s(self) -> float:
        return self._default_timeout_s

    def list_agents(self) -> list[AcpAgentConfig]:
        return list(self._agents.values())

    def resolve(self, name: str) -> AcpAgentConfig | None:
        agent = self._agents.get(name)
        return agent if agent is not None and agent.enabled else None


class AcpDelegateInput(BaseModel):
    agent: str
    task: str
    cwd: str


def _make_acp_delegate_handler(
    manager: AcpManager,
    on_chunk: Callable[[str, str], None] | None,
    path_sandbox: PathSandbox | None,
) -> ToolHandler:
    async def handler(tool_input: dict[str, Any]) -> str:
        try:
            parsed = AcpDelegateInput.model_validate(tool_input)
        except ValidationError as exc:
            raise ToolExecutionError(f"参数不合法：{exc}") from exc

        config = manager.resolve(parsed.agent)
        if config is None:
            raise ToolExecutionError(f'未找到已启用的 ACP agent "{parsed.agent}"')

        forward_chunk = (
            None if on_chunk is None else (lambda text: on_chunk(parsed.agent, text))
        )
        result = await run_acp_task(
            executable=config.executable,
            args=tuple(config.args),
            cwd=parsed.cwd,
            task=parsed.task,
            timeout_s=config.timeout_s or manager.default_timeout_s,
            on_chunk=forward_chunk,
            path_sandbox=path_sandbox,
        )
        payload = {
            "success": result.success,
            "content": result.content,
            "error": result.error,
            "stop_reason": result.stop_reason,
        }
        return json.dumps(payload, ensure_ascii=False)

    return handler


def register_acp_delegate_tool(
    manager: AcpManager,
    registry: ToolRegistry,
    on_chunk: Callable[[str, str], None] | None = None,
    *,
    path_sandbox: PathSandbox | None = None,
) -> None:
    agent_names = [agent.name for agent in manager.list_agents() if agent.enabled]
    if not agent_names:
        logger.info("未配置任何已启用的 ACP agent，跳过 acp_delegate 工具注册")
        return
    registry.register(
        ToolRegistration(
            definition=ToolDefinition(
                name=_ACP_DELEGATE_TOOL_NAME,
                description=(
                    "把一个复杂任务委派给本机已安装的外部编码 agent（如 Claude Code、Codex）独立"
                    "运行一整个 turn，适合需要该 agent 自带工具链（文件编辑、终端等）处理的任务。"
                    f"默认超时 {manager.default_timeout_s:.0f} 秒（可在设置里为单个 agent 单独"
                    f"配置），超时会取消任务并报告失败。可用的 agent：{', '.join(agent_names)}。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "enum": agent_names,
                            "description": "要委派给哪个已配置的外部 agent",
                        },
                        "task": {
                            "type": "string",
                            "description": "交给外部 agent 的完整任务描述",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "外部 agent 运行时的工作目录（必须是它需要操作的目录）",
                        },
                    },
                    "required": ["agent", "task", "cwd"],
                },
            ),
            handler=_make_acp_delegate_handler(manager, on_chunk, path_sandbox),
            policy_spec=ToolPolicySpec(path_arg="cwd"),
        )
    )
