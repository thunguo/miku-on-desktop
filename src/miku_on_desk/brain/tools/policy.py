"""工具执行前的纵深防御闸门。

七层判断按固定顺序执行，顺序本身就是安全属性，不是随意的实现细节：

1. 管理员显式禁用列表——绝对拒绝，任何后续层级都不能推翻。
2. 路径沙箱（结构性边界，见 ``path_sandbox.py``）。
3. 先读后改检查（结构性边界，见 ``read_tracker.py``）。
4. 工具自身的 ``requires_confirmation`` 标记。
5. 信任层：全局 trusted_mode / 用户允许列表 / 本会话已授权。
6. 危险命令正则（仅对声明了 ``command_arg`` 的工具生效）。
7. 兜底默认决策。

第 2、3 层特意放在第 4、5 层之前："这个工具是否已经拿到过许可"和"这条路径/命令本身是否安全"
必须是两个独立维度,信任层只能把原本会问用户的情形提升为直接放行，绝不能用来豁免路径沙箱或
先读后改这两条结构性边界——否则一次工具级别的信任授予会连带打穿沙箱本身。

``evaluate()`` 不区分工具来源：MCP 桥接工具（`mcp/host.py::_infer_policy_spec`）产出的
``ToolPolicySpec`` 和 builtin 工具手写的一样，走的是同一套判断顺序，没有为"这是 MCP 工具"
开任何后门。``McpServerConfig.trusted`` 也只能影响 spec 里的 ``requires_confirmation``
（第 4 层），改变不了第 2、3 层的结构性边界——一个被标记为可信的 MCP server，其工具的路径
参数依然要过 `path_sandbox`/`read_tracker`。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from miku_on_desk.brain.tools.path_sandbox import PathSandbox
from miku_on_desk.brain.tools.read_tracker import ReadTracker
from miku_on_desk.config.settings import PermissionsConfig

# 只识别单个 `;`/`<`/`>`/单个 `|`/单个 `&`，明确排除 `&&`/`||`——这两个是极其常见的
# 正常命令连接符，不排除会把大量正常操作误判成危险命令。
_SUSPICIOUS_SHELL_CHARS = re.compile(r"[;<>]|(?<!\|)\|(?!\|)|(?<!&)&(?!&)")


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyDecision:
    decision: Decision
    reason: str | None = None


@dataclass(frozen=True)
class ToolPolicySpec:
    """描述一个工具要走哪些额外闸门；由工具注册时提供，不出现在 LLM 可见的 schema 里。"""

    requires_confirmation: bool = False
    confirm_reason: str = "此操作需要用户确认。"
    command_arg: str | None = None
    """该工具输入里承载 shell 命令字符串的字段名，None 表示该工具不涉及命令执行。"""
    path_arg: str | None = None
    """该工具输入里承载文件路径的字段名，None 表示该工具不涉及路径沙箱。"""
    is_write: bool = False
    """是否为写操作,写操作需要先过 ``ReadTracker`` 的先读后改检查。"""


def is_dangerous_command(command: str) -> bool:
    return bool(_SUSPICIOUS_SHELL_CHARS.search(command))


@dataclass
class PolicyEngine:
    trusted_mode: bool
    allowed_tools: frozenset[str]
    denied_tools: frozenset[str]
    default_decision: Decision
    path_sandbox: PathSandbox
    read_tracker: ReadTracker
    _session_grants: dict[str, set[str]] = field(default_factory=dict, init=False)

    def grant(self, session_id: str, tool_name: str) -> None:
        """用户批准"本会话内不再询问这个工具"之后调用；不影响路径沙箱/先读后改这两条结构性边界。"""
        self._session_grants.setdefault(session_id, set()).add(tool_name)

    def clear_session(self, session_id: str) -> None:
        self._session_grants.pop(session_id, None)
        self.read_tracker.clear_session(session_id)

    def _is_granted(self, session_id: str, tool_name: str) -> bool:
        return tool_name in self._session_grants.get(session_id, set())

    def evaluate(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        spec: ToolPolicySpec,
        *,
        session_id: str,
    ) -> PolicyDecision:
        if tool_name in self.denied_tools:
            return PolicyDecision(Decision.DENY, f'工具 "{tool_name}" 已被管理员禁用。')

        if spec.path_arg is not None:
            raw_path = tool_input.get(spec.path_arg)
            if not isinstance(raw_path, str):
                return PolicyDecision(
                    Decision.DENY, f'工具 "{tool_name}" 缺少必需的路径参数 "{spec.path_arg}"。'
                )
            path = Path(raw_path)
            sandbox_result = self.path_sandbox.check(path)
            if not sandbox_result.allowed:
                return PolicyDecision(Decision.DENY, sandbox_result.reason)
            if spec.is_write and not self.read_tracker.has_been_read(session_id, path):
                return PolicyDecision(
                    Decision.DENY,
                    f'必须先用 read_file 读取过 "{raw_path}"，才能对它执行写操作——'
                    "不了解现有内容的盲写风险太高。",
                )

        # requires_confirmation 必须在会话授权缓存之前判断，否则一次会话级信任授予会
        # 连带跳过工具本身要求的确认。
        if spec.requires_confirmation and not self.trusted_mode:
            return PolicyDecision(Decision.ASK, spec.confirm_reason)

        if (
            self.trusted_mode
            or tool_name in self.allowed_tools
            or self._is_granted(session_id, tool_name)
        ):
            return PolicyDecision(Decision.ALLOW)

        if spec.command_arg is not None:
            command = str(tool_input.get(spec.command_arg, ""))
            if is_dangerous_command(command):
                return PolicyDecision(
                    Decision.ASK,
                    f'命令包含需要人工确认的字符（单个 ; < > | &）："{command}"',
                )

        return PolicyDecision(self.default_decision)


def default_policy_engine(
    config: PermissionsConfig, path_sandbox: PathSandbox, read_tracker: ReadTracker
) -> PolicyEngine:
    return PolicyEngine(
        trusted_mode=config.trusted_mode,
        allowed_tools=frozenset(config.allowed_tools),
        denied_tools=frozenset(config.denied_tools),
        default_decision=Decision(config.default_decision),
        path_sandbox=path_sandbox,
        read_tracker=read_tracker,
    )
