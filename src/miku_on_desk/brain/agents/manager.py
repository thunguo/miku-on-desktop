"""Sub-agent profile 管理。

延续 `secrets/vault.py` 的判断而不是 `memory/store.py` 的判断：agent profile 是低频 CRUD（用户
在设置面板里偶尔增删改一次，AI 循环运行时只读不写），不需要 `memory/store.py` 那种为高频写入
和 FTS5 全文检索设计的异步连接池，同步 `sqlite3` 更直接。

内置 profile（researcher/operator/planner）的系统提示词是本项目原创改写：直接引用
`read_file`/`grep`/`exec_command`/`webfetch` 等这个项目根本没有规划的工具会产生误导 agent
去调用不存在能力的提示词；这里只保留每个 profile 的行为契约（专注度、彻查程度、交叉验证、
证据优先、诚实汇报、结构化输出），不引用不存在的工具——包括 planner 不写依赖不存在的
`create_plan` 工具的交接语，planner 在这里就是把结构化计划作为最终文本返回，不做工具调用
交接。

同样不实现一个"记忆变化时反向使 frozen system 失效"的回调：理由与 `skills/manager.py` 顶部
说明一致，`frozen_system.py` 只接受上游拼好的字符串摘要，是否重建由组装 frozen system 的
上层逻辑决定。
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from miku_on_desk.config.settings import EnvBootstrap

_RESEARCHER_PROMPT = """你是一个只读调研 sub-agent。你的任务是彻查用户交代的问题，不做任何会
改变系统状态的操作。

要求：
- 专注于分配给你的具体任务，不要扩展到无关话题。
- 穷尽手头可用的信息来源，交叉验证关键结论，不要只采信单一来源就下结论。
- 报告中明确区分"已核实的事实"与"推测"，不确定的地方直接说不确定，不要编造细节掩盖信息缺口。
- 用结构化的方式输出结论：先给结论，再给支撑依据。"""

_OPERATOR_PROMPT = """你是一个可执行实际操作的 sub-agent。你可以使用当前授权范围内的所有工具
来完成分配给你的任务。

要求：
- 在执行有副作用的操作前，先确认自己理解的任务范围与用户意图一致。
- 每一步操作后检查其实际效果是否符合预期，不要假设操作一定成功。
- 如果任务在当前权限或工具能力下无法完成，明确报告卡在哪一步、原因是什么，不要假装完成。
- 最终报告要清楚说明实际做了什么、结果如何，不要只报告"已完成"这类空泛结论。"""

_PLANNER_PROMPT = """你是一个规划 sub-agent。你的任务是先调研清楚任务背景，再产出一份结构化的
可执行计划，你自己不执行计划中的步骤。

要求：
- 调研阶段要彻查任务涉及的现状，不要在信息不足的情况下直接开始规划。
- 计划要拆解到具体可执行的步骤，每一步说明要做什么、怎么做、如何验证这一步做对了。
- 诚实评估计划的风险点和复杂度，不要为了让计划看起来简单而隐藏已知的风险。
- 最终把计划作为一个 fenced ```json:plan``` 代码块返回，JSON 结构为：
  {"title": str, "goal": str, "success_criteria": [str], "steps": [{"title": str,
  "description": str, "approach": str, "verification": str}], "risks": [str],
  "estimated_complexity": "low" | "medium" | "high"}。
  代码块之外可以有简短的自然语言说明，但结构化计划本身必须是合法 JSON。"""


@dataclass(frozen=True)
class AgentProfile:
    id: str
    name: str
    description: str
    system_prompt: str
    tools: tuple[str, ...] = ()
    """派给这个 profile 的 sub-agent 能使用的工具白名单；空 tuple 表示"当前已注册的全部工具"。"""
    max_rounds: int = 20
    builtin: bool = False
    enabled: bool = True


@dataclass(frozen=True)
class _BuiltinSeed:
    id: str
    name: str
    description: str
    system_prompt: str
    tools: tuple[str, ...]
    max_rounds: int


_BUILTIN_SEEDS: tuple[_BuiltinSeed, ...] = (
    _BuiltinSeed(
        id="builtin_researcher",
        name="researcher",
        description="只读调研：彻查问题、交叉验证、给出有依据的结论，不执行任何有副作用的操作。",
        system_prompt=_RESEARCHER_PROMPT,
        tools=("skill",),
        max_rounds=20,
    ),
    _BuiltinSeed(
        id="builtin_operator",
        name="operator",
        description="可执行实际操作：在授权范围内使用全部可用工具完成具体任务。",
        system_prompt=_OPERATOR_PROMPT,
        tools=(),
        max_rounds=20,
    ),
    _BuiltinSeed(
        id="builtin_planner",
        name="planner",
        description="先调研后规划：产出结构化的可执行计划，自己不执行计划步骤。",
        system_prompt=_PLANNER_PROMPT,
        tools=("skill",),
        max_rounds=25,
    ),
)


def _row_to_profile(row: sqlite3.Row) -> AgentProfile:
    return AgentProfile(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        system_prompt=row["system_prompt"],
        tools=tuple(json.loads(row["tools"])),
        max_rounds=row["max_rounds"],
        builtin=bool(row["builtin"]),
        enabled=bool(row["enabled"]),
    )


@dataclass
class _UpdateSentinel:
    """区分"没传这个字段"和"传了 None"，因为 description/tools 等字段本身允许被设为空值。"""

    value: object = field(default=None)


_UNSET = _UpdateSentinel()


class AgentManager:
    """管理 sub-agent profile 的持久化 CRUD，以及内置 profile 的播种/解析。"""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_profiles ("
            "id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL, description TEXT NOT NULL, "
            "system_prompt TEXT NOT NULL, tools TEXT NOT NULL, max_rounds INTEGER NOT NULL, "
            "builtin INTEGER NOT NULL, enabled INTEGER NOT NULL, "
            "created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)"
        )
        self._conn.commit()
        self._seed_builtins()

    def _seed_builtins(self) -> None:
        now = int(time.time())
        for seed in _BUILTIN_SEEDS:
            self._conn.execute(
                "INSERT INTO agent_profiles (id, name, description, system_prompt, tools, "
                "max_rounds, builtin, enabled, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, 1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET description=excluded.description, "
                "system_prompt=excluded.system_prompt, tools=excluded.tools, "
                "max_rounds=excluded.max_rounds, updated_at=excluded.updated_at",
                (
                    seed.id,
                    seed.name,
                    seed.description,
                    seed.system_prompt,
                    json.dumps(list(seed.tools)),
                    seed.max_rounds,
                    now,
                    now,
                ),
            )
        builtin_ids = tuple(seed.id for seed in _BUILTIN_SEEDS)
        placeholders = ",".join("?" for _ in builtin_ids)
        self._conn.execute(
            f"DELETE FROM agent_profiles WHERE builtin = 1 AND id NOT IN ({placeholders})",
            builtin_ids,
        )
        self._conn.commit()

    def list_agents(self) -> list[AgentProfile]:
        rows = self._conn.execute("SELECT * FROM agent_profiles ORDER BY name").fetchall()
        return [_row_to_profile(row) for row in rows]

    def get_agent(self, agent_id: str) -> AgentProfile | None:
        row = self._conn.execute(
            "SELECT * FROM agent_profiles WHERE id = ?", (agent_id,)
        ).fetchone()
        return _row_to_profile(row) if row is not None else None

    def resolve_profile(self, name_or_id: str) -> AgentProfile | None:
        row = self._conn.execute(
            "SELECT * FROM agent_profiles WHERE (name = ? OR id = ?) AND enabled = 1",
            (name_or_id, name_or_id),
        ).fetchone()
        return _row_to_profile(row) if row is not None else None

    def create_agent(
        self,
        *,
        name: str,
        description: str,
        system_prompt: str,
        tools: tuple[str, ...] = (),
        max_rounds: int = 20,
    ) -> AgentProfile:
        if not name.strip():
            raise ValueError("name 不能为空")
        if not system_prompt.strip():
            raise ValueError("system_prompt 不能为空")
        if max_rounds <= 0:
            raise ValueError("max_rounds 必须大于 0")
        agent_id = f"agent_{uuid.uuid4().hex}"
        now = int(time.time())
        try:
            self._conn.execute(
                "INSERT INTO agent_profiles (id, name, description, system_prompt, tools, "
                "max_rounds, builtin, enabled, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, 1, ?, ?)",
                (
                    agent_id,
                    name,
                    description,
                    system_prompt,
                    json.dumps(list(tools)),
                    max_rounds,
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f'已存在名为 "{name}" 的 agent profile') from exc
        self._conn.commit()
        profile = self.get_agent(agent_id)
        assert profile is not None
        return profile

    def update_agent(
        self,
        agent_id: str,
        *,
        name: str | _UpdateSentinel = _UNSET,
        description: str | _UpdateSentinel = _UNSET,
        system_prompt: str | _UpdateSentinel = _UNSET,
        tools: tuple[str, ...] | _UpdateSentinel = _UNSET,
        max_rounds: int | _UpdateSentinel = _UNSET,
        enabled: bool | _UpdateSentinel = _UNSET,
    ) -> AgentProfile:
        existing = self.get_agent(agent_id)
        if existing is None:
            raise ValueError(f'未找到 agent profile "{agent_id}"')
        if existing.builtin and not isinstance(name, _UpdateSentinel):
            raise ValueError("内置 agent profile 不允许改名")

        next_name = existing.name if isinstance(name, _UpdateSentinel) else name
        next_description = (
            existing.description if isinstance(description, _UpdateSentinel) else description
        )
        next_prompt = (
            existing.system_prompt
            if isinstance(system_prompt, _UpdateSentinel)
            else system_prompt
        )
        next_tools = existing.tools if isinstance(tools, _UpdateSentinel) else tools
        next_max_rounds = (
            existing.max_rounds if isinstance(max_rounds, _UpdateSentinel) else max_rounds
        )
        next_enabled = existing.enabled if isinstance(enabled, _UpdateSentinel) else enabled

        if not next_name.strip():
            raise ValueError("name 不能为空")
        if not next_prompt.strip():
            raise ValueError("system_prompt 不能为空")
        if next_max_rounds <= 0:
            raise ValueError("max_rounds 必须大于 0")

        try:
            self._conn.execute(
                "UPDATE agent_profiles SET name = ?, description = ?, system_prompt = ?, "
                "tools = ?, max_rounds = ?, enabled = ?, updated_at = ? WHERE id = ?",
                (
                    next_name,
                    next_description,
                    next_prompt,
                    json.dumps(list(next_tools)),
                    next_max_rounds,
                    int(next_enabled),
                    int(time.time()),
                    agent_id,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f'已存在名为 "{next_name}" 的 agent profile') from exc
        self._conn.commit()
        profile = self.get_agent(agent_id)
        assert profile is not None
        return profile

    def delete_agent(self, agent_id: str) -> None:
        existing = self.get_agent(agent_id)
        if existing is None:
            raise ValueError(f'未找到 agent profile "{agent_id}"')
        if existing.builtin:
            raise ValueError("内置 agent profile 不允许删除，如需恢复默认值请使用 reset")
        self._conn.execute("DELETE FROM agent_profiles WHERE id = ?", (agent_id,))
        self._conn.commit()

    def reset_builtin_agent(self, agent_id: str) -> AgentProfile:
        seed = next((s for s in _BUILTIN_SEEDS if s.id == agent_id), None)
        if seed is None:
            raise ValueError(f'"{agent_id}" 不是内置 agent profile')
        now = int(time.time())
        self._conn.execute(
            "UPDATE agent_profiles SET description = ?, system_prompt = ?, tools = ?, "
            "max_rounds = ?, enabled = 1, updated_at = ? WHERE id = ?",
            (
                seed.description,
                seed.system_prompt,
                json.dumps(list(seed.tools)),
                seed.max_rounds,
                now,
                agent_id,
            ),
        )
        self._conn.commit()
        profile = self.get_agent(agent_id)
        assert profile is not None
        return profile

    def close(self) -> None:
        self._conn.close()


def default_agent_db_path(bootstrap: EnvBootstrap | None = None) -> Path:
    bootstrap = bootstrap or EnvBootstrap()
    return bootstrap.resolve_data_dir() / "agents.db"


def default_agent_manager(bootstrap: EnvBootstrap | None = None) -> AgentManager:
    return AgentManager(default_agent_db_path(bootstrap))
