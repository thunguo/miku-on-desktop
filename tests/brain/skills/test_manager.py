"""SkillManager 的回归测试：真实 tmp_path 目录 + 真实文件读写，不 mock 文件系统。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from miku_on_desk.brain.providers.base import ToolUseBlock
from miku_on_desk.brain.skills.manager import (
    SkillManager,
    SkillValidationError,
    description_ok,
    name_ok,
    parse_skill_markdown,
    register_skill_tool,
    validate_skill_markdown_content,
)
from miku_on_desk.brain.tools.path_sandbox import PathSandbox
from miku_on_desk.brain.tools.policy import Decision, PolicyEngine
from miku_on_desk.brain.tools.read_tracker import ReadTracker
from miku_on_desk.brain.tools.registry import ToolRegistry


def _write_skill(
    skills_dir: Path, name: str, *, frontmatter_name: str | None = None, body: str = "body"
) -> None:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    fm_name = frontmatter_name if frontmatter_name is not None else name
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {fm_name}\ndescription: does {name}ing\n---\n\n{body}\n", encoding="utf-8"
    )


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


def test_name_ok_accepts_kebab_case_and_rejects_others() -> None:
    assert name_ok("fooer") is True
    assert name_ok("foo-bar-2") is True
    assert name_ok("") is False
    assert name_ok("Foo") is False
    assert name_ok("foo_bar") is False
    assert name_ok("-foo") is False
    assert name_ok("a" * 65) is False


def test_description_ok_rejects_angle_brackets_and_empty() -> None:
    assert description_ok("does fooing") is True
    assert description_ok("") is False
    assert description_ok("has <tag>") is False
    assert description_ok("x" * 1025) is False


def test_parse_skill_markdown_extracts_frontmatter_fields() -> None:
    parsed = parse_skill_markdown("---\nname: fooer\ndescription: does fooing\n---\n\nbody text")

    assert parsed.name == "fooer"
    assert parsed.description == "does fooing"


def test_validate_skill_markdown_content_rejects_invalid_name() -> None:
    with pytest.raises(SkillValidationError, match="name 不合法"):
        validate_skill_markdown_content("---\nname: Bad_Name\ndescription: x\n---\n")


def test_validate_skill_markdown_content_rejects_empty_content() -> None:
    with pytest.raises(SkillValidationError, match="不能为空"):
        validate_skill_markdown_content("   ")


def test_list_discovered_skills_returns_valid_skill_with_parsed_description(tmp_path: Path) -> None:
    _write_skill(tmp_path, "fooer")
    manager = SkillManager(tmp_path)

    skills = manager.list_discovered_skills()

    assert [s.name for s in skills] == ["fooer"]
    assert skills[0].description == "does fooering"
    assert skills[0].tainted is False


def test_list_discovered_skills_excludes_missing_skill_md(tmp_path: Path) -> None:
    (tmp_path / "empty-dir").mkdir()
    manager = SkillManager(tmp_path)

    assert manager.list_discovered_skills() == []
    all_skills = manager.list_all_skills()
    assert all_skills[0].tainted is True
    assert all_skills[0].taint_reason is not None
    assert "缺少 SKILL.md" in all_skills[0].taint_reason


def test_list_discovered_skills_excludes_frontmatter_name_mismatch(tmp_path: Path) -> None:
    _write_skill(tmp_path, "fooer", frontmatter_name="wrong-name")
    manager = SkillManager(tmp_path)

    assert manager.list_discovered_skills() == []
    all_skills = manager.list_all_skills()
    assert all_skills[0].tainted is True
    assert all_skills[0].taint_reason is not None
    assert "frontmatter name" in all_skills[0].taint_reason


def test_get_skill_returns_none_for_tainted_or_unknown(tmp_path: Path) -> None:
    _write_skill(tmp_path, "fooer", frontmatter_name="wrong-name")
    manager = SkillManager(tmp_path)

    assert manager.get_skill("fooer") is None
    assert manager.get_skill("does-not-exist") is None


def test_load_skill_content_returns_full_markdown_text(tmp_path: Path) -> None:
    _write_skill(tmp_path, "fooer", body="step 1: do the thing")
    manager = SkillManager(tmp_path)

    content = manager.load_skill_content("fooer")

    assert content is not None
    assert "step 1: do the thing" in content
    assert "name: fooer" in content


def test_load_skill_content_returns_none_for_tainted_skill(tmp_path: Path) -> None:
    (tmp_path / "empty-dir").mkdir()
    manager = SkillManager(tmp_path)

    assert manager.load_skill_content("empty-dir") is None


def test_manifest_is_cached_until_invalidated(tmp_path: Path) -> None:
    manager = SkillManager(tmp_path)
    assert manager.list_discovered_skills() == []

    _write_skill(tmp_path, "fooer")
    assert manager.list_discovered_skills() == []

    manager.invalidate()
    assert [s.name for s in manager.list_discovered_skills()] == ["fooer"]


def test_build_prompt_section_returns_empty_string_when_no_skills(tmp_path: Path) -> None:
    manager = SkillManager(tmp_path)

    assert manager.build_prompt_section() == ""


def test_build_prompt_section_lists_available_skills_as_xml(tmp_path: Path) -> None:
    _write_skill(tmp_path, "fooer")
    manager = SkillManager(tmp_path)

    section = manager.build_prompt_section()

    assert "<available_skills>" in section
    assert "<name>fooer</name>" in section
    assert "<description>does fooering</description>" in section


async def test_watch_invalidates_manifest_on_new_skill_file(tmp_path: Path) -> None:
    manager = SkillManager(tmp_path)
    assert manager.list_discovered_skills() == []

    watch_task = asyncio.create_task(manager.watch())
    try:
        await asyncio.sleep(0.2)  # 给 watchfiles 的 inotify/fsevents watcher 留出启动时间
        _write_skill(tmp_path, "fooer")

        for _ in range(50):
            if manager.list_discovered_skills():
                break
            await asyncio.sleep(0.1)
    finally:
        watch_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await watch_task

    assert [s.name for s in manager.list_discovered_skills()] == ["fooer"]


async def test_register_skill_tool_executes_through_to_manager(
    tmp_path: Path, registry: ToolRegistry
) -> None:
    _write_skill(tmp_path, "fooer", body="do the fooing")
    manager = SkillManager(tmp_path)
    register_skill_tool(manager, registry)

    result = await registry.execute(
        ToolUseBlock(id="call1", name="skill", input={"name": "fooer"}), session_id="s1"
    )

    assert result.is_error is False
    assert "do the fooing" in result.content


async def test_register_skill_tool_reports_error_for_unknown_skill(
    tmp_path: Path, registry: ToolRegistry
) -> None:
    manager = SkillManager(tmp_path)
    register_skill_tool(manager, registry)

    result = await registry.execute(
        ToolUseBlock(id="call1", name="skill", input={"name": "does-not-exist"}), session_id="s1"
    )

    assert result.is_error is True
