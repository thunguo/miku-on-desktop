"""build_frozen_system 的拼接顺序与空分区跳过逻辑回归测试。"""

from __future__ import annotations

from miku_on_desk.brain.prompt.frozen_system import FrozenSystemSections, build_frozen_system


def test_identity_only_when_all_sections_empty() -> None:
    sections = FrozenSystemSections(identity="你是初音未来。")
    assert build_frozen_system(sections) == "你是初音未来。"


def test_sections_appended_in_ascending_change_frequency_order() -> None:
    sections = FrozenSystemSections(
        identity="身份",
        core_memory="核心记忆内容",
        skills_summary="技能摘要",
        agents_summary="子代理摘要",
        memory_index_summary="记忆索引内容",
    )
    result = build_frozen_system(sections)
    assert result == (
        "身份\n\n"
        "## 已启用的 Sub-agent\n\n子代理摘要\n\n"
        "## 已启用的 Skills\n\n技能摘要\n\n"
        "## 记忆索引\n\n记忆索引内容\n\n"
        "## 核心记忆\n\n核心记忆内容"
    )


def test_missing_sections_are_skipped_without_leaving_gaps() -> None:
    sections = FrozenSystemSections(identity="身份", core_memory="核心记忆内容")
    result = build_frozen_system(sections)
    assert result == "身份\n\n## 核心记忆\n\n核心记忆内容"
    assert "Sub-agent" not in result
    assert "Skills" not in result
