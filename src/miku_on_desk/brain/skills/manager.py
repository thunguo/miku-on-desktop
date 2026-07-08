"""Skill 加载/校验/热重载。

`skills_dir` 是本机磁盘上的一个目录，manifest 扫描直接用 `pathlib` 读取；`get_manifest`/
`list_*`/`get_skill`/`load_skill_content` 都是同步方法。

AI 驱动的 skill 写入能力（创建/编辑/patch/删除用户 skill）暂不实现：这本应是内置文件操作工具
（`write_file`/`edit_file`/`multi_edit`）里"检测到目标路径落在 skills 目录下就额外做一次
frontmatter 校验"的旁路逻辑，而这几个内置文件操作工具本身在本项目里还未实现。等它们落地时
再把对应校验路径接进来，不为一个还不存在的调用方预先搭好整套 CRUD。

manifest 缓存失效直接用 `watchfiles.awatch` 监听本机目录变化，不需要额外的事件转发层。

不做"文件变化连带使冻结系统提示重新构建"这层反向依赖：`frozen_system.py` 明确只接受上游子
系统产出的纯字符串摘要（见其模块 docstring），是否需要重建、何时重建由组装 frozen system 的
上层逻辑决定，`SkillManager` 只负责如实反映"当前 manifest 是否已失效"，不主动去 poke 别的
模块。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter
from watchfiles import awatch

from miku_on_desk.brain.providers.base import ToolDefinition
from miku_on_desk.brain.tools.registry import (
    ToolExecutionError,
    ToolHandler,
    ToolRegistration,
    ToolRegistry,
)
from miku_on_desk.config.settings import EnvBootstrap

logger = logging.getLogger(__name__)

_SKILL_FILENAME = "SKILL.md"
_MAX_SKILL_MD_BYTES = 64 * 1024
_MAX_PROMPT_INJECT = 80
_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class SkillValidationError(Exception):
    """SKILL.md 内容或 frontmatter 未通过校验。"""


def name_ok(name: str) -> bool:
    name = name.strip()
    return bool(name) and len(name) <= 64 and bool(_NAME_RE.fullmatch(name))


def description_ok(description: str) -> bool:
    description = description.strip()
    if not description or len(description) > 1024:
        return False
    return "<" not in description and ">" not in description


@dataclass(frozen=True)
class ParsedSkillMarkdown:
    name: str
    description: str
    content: str


def parse_skill_markdown(raw: str) -> ParsedSkillMarkdown:
    post = frontmatter.loads(raw)
    name = str(post.metadata.get("name", "")).strip()
    description = str(post.metadata.get("description", "")).strip()
    return ParsedSkillMarkdown(name=name, description=description, content=raw.rstrip())


def validate_skill_markdown_content(content: str) -> ParsedSkillMarkdown:
    if not content.strip():
        raise SkillValidationError("content 不能为空")
    if len(content.encode("utf-8")) > _MAX_SKILL_MD_BYTES:
        raise SkillValidationError(f"content 超过 {_MAX_SKILL_MD_BYTES // 1024}KB 上限")
    try:
        parsed = parse_skill_markdown(content)
    except Exception as exc:
        raise SkillValidationError(f"解析 YAML frontmatter 失败：{exc}") from exc
    if not name_ok(parsed.name):
        raise SkillValidationError(
            f'frontmatter name 不合法："{parsed.name}"'
            "（须匹配 ^[a-z0-9]+(-[a-z0-9]+)*$，最长 64 字符）"
        )
    if not description_ok(parsed.description):
        raise SkillValidationError(
            "frontmatter description 不合法（1-1024 字符，不能包含 '<' 或 '>'）"
        )
    return parsed


@dataclass(frozen=True)
class SkillEntry:
    name: str
    description: str
    dir: Path
    location: Path | None = None
    tainted: bool = False
    taint_reason: str | None = None


@dataclass(frozen=True)
class SkillManifest:
    root: Path
    skills_by_name: dict[str, SkillEntry]


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SkillManager:
    """管理单个 `skills_dir` 下所有 skill 的 manifest 缓存、校验与查询。"""

    def __init__(self, skills_dir: Path) -> None:
        self._skills_dir = skills_dir
        self._manifest: SkillManifest | None = None

    @property
    def skills_dir(self) -> Path:
        return self._skills_dir

    def invalidate(self) -> None:
        self._manifest = None

    def get_manifest(self) -> SkillManifest:
        if self._manifest is None:
            self._manifest = self._build_manifest()
        return self._manifest

    def _build_manifest(self) -> SkillManifest:
        skills_by_name: dict[str, SkillEntry] = {}
        if not self._skills_dir.is_dir():
            return SkillManifest(root=self._skills_dir, skills_by_name=skills_by_name)

        for entry_dir in sorted(p for p in self._skills_dir.iterdir() if p.is_dir()):
            name = entry_dir.name
            skills_by_name[name] = self._load_entry(name, entry_dir)
        return SkillManifest(root=self._skills_dir, skills_by_name=skills_by_name)

    def _load_entry(self, name: str, entry_dir: Path) -> SkillEntry:
        skill_md = entry_dir / _SKILL_FILENAME
        if not skill_md.is_file():
            return SkillEntry(
                name=name, description="", dir=entry_dir, tainted=True, taint_reason="缺少 SKILL.md"
            )

        try:
            raw = skill_md.read_text(encoding="utf-8")
        except OSError as exc:
            return SkillEntry(
                name=name,
                description="",
                dir=entry_dir,
                tainted=True,
                taint_reason=f"读取失败：{exc}",
            )

        try:
            parsed = validate_skill_markdown_content(raw)
        except SkillValidationError as exc:
            return SkillEntry(
                name=name, description="", dir=entry_dir, tainted=True, taint_reason=str(exc)
            )

        if parsed.name != name:
            return SkillEntry(
                name=name,
                description=parsed.description,
                dir=entry_dir,
                tainted=True,
                taint_reason=f'frontmatter name "{parsed.name}" 与目录名 "{name}" 不一致',
            )

        return SkillEntry(
            name=parsed.name,
            description=parsed.description,
            dir=entry_dir,
            location=skill_md,
            tainted=False,
        )

    def list_discovered_skills(self) -> list[SkillEntry]:
        manifest = self.get_manifest()
        return sorted(
            (s for s in manifest.skills_by_name.values() if not s.tainted), key=lambda s: s.name
        )

    def list_all_skills(self) -> list[SkillEntry]:
        manifest = self.get_manifest()
        return sorted(manifest.skills_by_name.values(), key=lambda s: s.name)

    def get_skill(self, name: str) -> SkillEntry | None:
        if not name_ok(name):
            return None
        entry = self.get_manifest().skills_by_name.get(name)
        if entry is None or entry.tainted:
            return None
        return entry

    def load_skill_content(self, name: str) -> str | None:
        entry = self.get_skill(name)
        if entry is None or entry.location is None:
            return None
        try:
            return entry.location.read_text(encoding="utf-8").rstrip()
        except OSError:
            logger.warning('读取 skill "%s" 内容失败', name, exc_info=True)
            return None

    def build_prompt_section(self) -> str:
        discovered = self.list_discovered_skills()
        if not discovered:
            return ""

        lines = [
            f"可复用方法沉淀为 `{_SKILL_FILENAME}` 文件，位于 `{self._skills_dir}/<name>/`。",
            "用 `skill` 工具加载后严格按其中说明执行。",
            "当用户任务与某个 skill 的描述匹配时，优先使用该 skill 而非临时推理。",
            "",
            "<available_skills>",
        ]
        if len(discovered) > _MAX_PROMPT_INJECT:
            logger.warning(
                "有 %d 个 skill 因超过注入上限（%d）被截断",
                len(discovered) - _MAX_PROMPT_INJECT,
                _MAX_PROMPT_INJECT,
            )
        for skill in discovered[:_MAX_PROMPT_INJECT]:
            lines.append("  <skill>")
            lines.append(f"    <name>{_xml_escape(skill.name)}</name>")
            lines.append(f"    <description>{_xml_escape(skill.description)}</description>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    async def watch(self) -> None:
        async for _changes in awatch(self._skills_dir):
            self.invalidate()


def _make_skill_handler(manager: SkillManager) -> ToolHandler:
    async def handler(tool_input: dict[str, Any]) -> str:
        name = str(tool_input.get("name", "")).strip()
        content = manager.load_skill_content(name)
        if content is None:
            raise ToolExecutionError(f'未找到名为 "{name}" 的 skill。')
        return content

    return handler


def register_skill_tool(manager: SkillManager, registry: ToolRegistry) -> None:
    registry.register(
        ToolRegistration(
            definition=ToolDefinition(
                name="skill",
                description=(
                    "加载一个 skill 的完整内容，并按其中的说明严格执行。"
                    "name 取自 <available_skills>。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "Skill 名称"}},
                    "required": ["name"],
                },
            ),
            handler=_make_skill_handler(manager),
        )
    )


def default_skill_manager(
    skills_dir: Path | None, bootstrap: EnvBootstrap | None = None
) -> SkillManager:
    bootstrap = bootstrap or EnvBootstrap()
    resolved = skills_dir if skills_dir is not None else bootstrap.resolve_data_dir() / "skills"
    resolved.mkdir(parents=True, exist_ok=True)
    return SkillManager(resolved)
