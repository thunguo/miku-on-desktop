"""AgentManager 的回归测试：真实 tmp_path 下的 sqlite 文件，不 mock 数据库。"""

from __future__ import annotations

from pathlib import Path

import pytest

from miku_on_desk.brain.agents.manager import AgentManager


@pytest.fixture
def manager(tmp_path: Path) -> AgentManager:
    return AgentManager(tmp_path / "agents.db")


def test_seeds_three_builtin_profiles_on_first_init(manager: AgentManager) -> None:
    names = {a.name for a in manager.list_agents()}
    assert names == {"researcher", "operator", "planner"}
    assert all(a.builtin for a in manager.list_agents())


def test_researcher_profile_is_read_only_whitelisted(manager: AgentManager) -> None:
    profile = manager.resolve_profile("researcher")
    assert profile is not None
    assert profile.tools == ("skill",)


def test_operator_profile_allows_all_tools_via_empty_whitelist(manager: AgentManager) -> None:
    profile = manager.resolve_profile("operator")
    assert profile is not None
    assert profile.tools == ()


def test_resolve_profile_matches_by_id_too(manager: AgentManager) -> None:
    by_name = manager.resolve_profile("planner")
    assert by_name is not None
    by_id = manager.resolve_profile(by_name.id)
    assert by_id == by_name


def test_resolve_profile_returns_none_for_unknown_name(manager: AgentManager) -> None:
    assert manager.resolve_profile("does-not-exist") is None


def test_create_agent_persists_and_is_retrievable(manager: AgentManager) -> None:
    created = manager.create_agent(
        name="custom", description="d", system_prompt="p", tools=("skill",), max_rounds=5
    )

    assert created.builtin is False
    fetched = manager.get_agent(created.id)
    assert fetched == created


def test_create_agent_rejects_empty_name(manager: AgentManager) -> None:
    with pytest.raises(ValueError, match="name 不能为空"):
        manager.create_agent(name="  ", description="d", system_prompt="p")


def test_create_agent_rejects_empty_system_prompt(manager: AgentManager) -> None:
    with pytest.raises(ValueError, match="system_prompt 不能为空"):
        manager.create_agent(name="x", description="d", system_prompt="  ")


def test_create_agent_rejects_non_positive_max_rounds(manager: AgentManager) -> None:
    with pytest.raises(ValueError, match="max_rounds 必须大于 0"):
        manager.create_agent(name="x", description="d", system_prompt="p", max_rounds=0)


def test_create_agent_rejects_duplicate_name(manager: AgentManager) -> None:
    manager.create_agent(name="dup", description="d", system_prompt="p")

    with pytest.raises(ValueError, match="已存在名为"):
        manager.create_agent(name="dup", description="d2", system_prompt="p2")


def test_update_agent_changes_fields(manager: AgentManager) -> None:
    created = manager.create_agent(name="x", description="d", system_prompt="p")

    updated = manager.update_agent(created.id, description="d2", max_rounds=7)

    assert updated.description == "d2"
    assert updated.max_rounds == 7
    assert updated.name == "x"


def test_update_agent_rejects_renaming_builtin(manager: AgentManager) -> None:
    researcher = manager.resolve_profile("researcher")
    assert researcher is not None

    with pytest.raises(ValueError, match="不允许改名"):
        manager.update_agent(researcher.id, name="renamed")


def test_update_agent_can_edit_builtin_non_name_fields(manager: AgentManager) -> None:
    researcher = manager.resolve_profile("researcher")
    assert researcher is not None

    updated = manager.update_agent(researcher.id, enabled=False)

    assert updated.enabled is False
    assert manager.resolve_profile("researcher") is None


def test_update_agent_raises_for_unknown_id(manager: AgentManager) -> None:
    with pytest.raises(ValueError, match="未找到"):
        manager.update_agent("does-not-exist", description="d")


def test_delete_agent_removes_custom_profile(manager: AgentManager) -> None:
    created = manager.create_agent(name="x", description="d", system_prompt="p")

    manager.delete_agent(created.id)

    assert manager.get_agent(created.id) is None


def test_delete_agent_rejects_builtin(manager: AgentManager) -> None:
    researcher = manager.resolve_profile("researcher")
    assert researcher is not None

    with pytest.raises(ValueError, match="不允许删除"):
        manager.delete_agent(researcher.id)


def test_reset_builtin_agent_restores_defaults(manager: AgentManager) -> None:
    researcher = manager.resolve_profile("researcher")
    assert researcher is not None
    manager.update_agent(researcher.id, max_rounds=99, enabled=False)

    restored = manager.reset_builtin_agent(researcher.id)

    assert restored.max_rounds == researcher.max_rounds
    assert restored.enabled is True


def test_reset_builtin_agent_rejects_non_builtin_id(manager: AgentManager) -> None:
    created = manager.create_agent(name="x", description="d", system_prompt="p")

    with pytest.raises(ValueError, match="不是内置"):
        manager.reset_builtin_agent(created.id)


def test_builtin_profiles_survive_reopening_the_same_db(tmp_path: Path) -> None:
    db_path = tmp_path / "agents.db"
    first = AgentManager(db_path)
    first.close()

    second = AgentManager(db_path)
    names = {a.name for a in second.list_agents()}

    assert names == {"researcher", "operator", "planner"}
