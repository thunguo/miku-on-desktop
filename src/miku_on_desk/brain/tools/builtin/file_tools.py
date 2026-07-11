"""read_file/write_file：激活项目已建好但从未被使用的路径沙箱 + 先读后写基础设施。

沙箱越权/未读先写的拒绝逻辑完全在 ``PolicyEngine`` 里，这里只负责真正的文件 IO。
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
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

_MAX_READ_FILE_CHARS = 40_000


class ReadFileInput(BaseModel):
    path: str


class WriteFileInput(BaseModel):
    path: str
    content: str


def _make_read_file_handler() -> ToolHandler:
    async def handler(tool_input: dict[str, Any]) -> str:
        try:
            parsed = ReadFileInput.model_validate(tool_input)
        except ValidationError as exc:
            raise ToolExecutionError(f"参数不合法：{exc}") from exc

        path = Path(parsed.path)
        if not path.exists():
            return f'文件 "{parsed.path}" 不存在。如果是要新建它，可以直接调用 write_file。'
        if path.is_dir():
            raise ToolExecutionError(
                f'"{parsed.path}" 是一个目录，read_file 只能读文件；'
                "查看目录内容可以用 exec_command 配合 ls。"
            )
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ToolExecutionError(f'读取 "{parsed.path}" 失败：{exc}') from exc

        if len(text) > _MAX_READ_FILE_CHARS:
            omitted = len(text) - _MAX_READ_FILE_CHARS
            return (
                text[:_MAX_READ_FILE_CHARS]
                + f"\n\n...[已截断，文件共 {len(text)} 字符，省略了后面 {omitted} 字符]"
            )
        return text

    return handler


def _make_write_file_handler() -> ToolHandler:
    async def handler(tool_input: dict[str, Any]) -> str:
        try:
            parsed = WriteFileInput.model_validate(tool_input)
        except ValidationError as exc:
            raise ToolExecutionError(f"参数不合法：{exc}") from exc

        path = Path(parsed.path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
            tmp_path.write_text(parsed.content, encoding="utf-8")
            os.replace(tmp_path, path)
        except OSError as exc:
            raise ToolExecutionError(f'写入 "{parsed.path}" 失败：{exc}') from exc

        return f'已写入 "{parsed.path}"（{len(parsed.content)} 字符）。'

    return handler


def register_file_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolRegistration(
            definition=ToolDefinition(
                name="read_file",
                description=(
                    "读取一个本地文件的完整内容。路径必须落在允许的目录范围内"
                    "（工作目录/输出目录/数据目录/用户常见工作目录等）。文件不存在不算错误——"
                    "如果确实要新建它，读过一次之后就可以直接调用 write_file。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "要读取的文件路径"},
                    },
                    "required": ["path"],
                },
            ),
            handler=_make_read_file_handler(),
            policy_spec=ToolPolicySpec(path_arg="path"),
            marks_read=True,
        )
    )
    registry.register(
        ToolRegistration(
            definition=ToolDefinition(
                name="write_file",
                description=(
                    "整体覆盖写入（或新建）一个本地文件，不支持局部编辑。必须先用 read_file "
                    "读过同一路径才能写——避免在不了解现有内容的情况下盲写覆盖。每次写入都需要"
                    "用户二次确认。"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "要写入的文件路径"},
                        "content": {"type": "string", "description": "写入的完整文件内容"},
                    },
                    "required": ["path", "content"],
                },
            ),
            handler=_make_write_file_handler(),
            policy_spec=ToolPolicySpec(
                path_arg="path",
                is_write=True,
                requires_confirmation=True,
                confirm_reason="即将写入本地文件（可能覆盖已有内容），是否允许？",
            ),
        )
    )
