"""把本地 hook sidecar 接入 Claude Code / Codex CLI / Gemini CLI 各自的配置文件。

Claude Code 的 ``http`` 类型 hook 是官方支持的配置方式：会同步 POST 事件 JSON 到给定
URL 并等待响应（默认超时 600s），不需要用户自己写 shell 脚本转发。

只接入纯通知性质的事件（``_MANAGED_EVENTS``）：``PreToolUse``/``PermissionRequest``/
``PermissionDenied`` 的 HTTP 响应体可能被 Claude Code 当作真正的允许/拒绝决策使用
（这点在公开文档里未被百分百确认），而本项目的 sidecar 只是想要视觉反馈、不追求协议
对接，所以这三个事件放进 ``_EXPERIMENTAL_EVENTS``，需要显式 opt-in（
``include_experimental=True``），且启用前必须对照当时最新的官方文档重新确认响应体
语义，否则错误的响应可能意外拦下真实的工具调用。

Codex CLI 与 Gemini CLI 目前都**只支持本地 ``command`` 类型 hook、JSON 走 stdin**，
没有 Claude Code 这种可以直接填 URL 的 ``http`` 类型，所以这两家改用
``build_forward_command`` 生成一条指向 ``face/hooks/forward.py``（console script
``miku-on-desk-hook-forward``）的命令，由它读 stdin 转发给同一个 sidecar。因为这个
转发程序自己保证"永远退出码 0、stdout 保持空"，不会被两家 CLI 的"解析 stdout JSON
作为决策"语义误读，所以理论上比 Claude Code 更安全；但下面几点仍是根据官方文档做的
合理推断、未在真实 CLI 环境里跑过，接入时需要重新核实：
Codex 的 ``hooks.json`` 文件路径推断为与 ``config.toml`` 同目录（``~/.codex/hooks.json``），
其顶层结构是否真的直接以事件名做 key（而不是再包一层）没有官方示例文件可核对；
Gemini CLI 的 lifecycle 事件（如 ``SessionStart``）matcher 是否支持空字符串通配全部
子类型也未被文档明确确认。
"""

from __future__ import annotations

import copy
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MANAGED_EVENTS = (
    "SessionStart",
    "UserPromptSubmit",
    "PostToolUse",
    "PostToolUseFailure",
    "Stop",
    "StopFailure",
    "SessionEnd",
)
_EXPERIMENTAL_EVENTS = ("PreToolUse", "PermissionRequest", "PermissionDenied")

_HOOK_TIMEOUT_SECONDS = 5

_FORWARD_COMMAND_NAME = "miku-on-desk-hook-forward"

# Codex 目前只在这几个事件上确认过 payload 字段（见 forward.py 模块文档），且它们都是
# 纯通知、不涉及允许/拒绝决策；PreToolUse/PermissionRequest 与 Claude Code 一样归入实验性。
_MANAGED_EVENTS_CODEX = ("SessionStart", "UserPromptSubmit", "PostToolUse", "Stop")
_EXPERIMENTAL_EVENTS_CODEX = ("PreToolUse", "PermissionRequest")

_MANAGED_EVENTS_GEMINI = ("SessionStart", "SessionEnd", "AfterTool", "AfterAgent")
_EXPERIMENTAL_EVENTS_GEMINI = ("BeforeTool", "BeforeAgent", "Notification")


@dataclass(frozen=True)
class InstallResult:
    added_events: tuple[str, ...]
    updated_events: tuple[str, ...]


def default_claude_settings_path(*, project_dir: Path | None = None) -> Path:
    """``project_dir`` 给定 → ``<project_dir>/.claude/settings.json``；否则 →
    ``~/.claude/settings.json``（默认用户级：桌宠是机器级常驻进程，不属于单个项目）。
    """
    base = project_dir if project_dir is not None else Path.home()
    return base / ".claude" / "settings.json"


def _build_hook_group(url: str, token: str) -> dict[str, Any]:
    return {
        "matcher": "",
        "hooks": [
            {
                "type": "http",
                "url": url,
                "headers": {"Authorization": f"Bearer {token}"},
                "timeout": _HOOK_TIMEOUT_SECONDS,
            }
        ],
    }


def _find_managed_hook(groups: list[Any], url: str) -> dict[str, Any] | None:
    for group in groups:
        for hook in group.get("hooks", []):
            if hook.get("type") == "http" and hook.get("url") == url:
                return hook  # type: ignore[no-any-return]
    return None


def merge_hooks(
    existing: dict[str, Any], *, url: str, token: str, include_experimental: bool
) -> tuple[dict[str, Any], InstallResult]:
    """纯函数，不做 I/O。用 ``(event, url)`` 匹配识别"这是不是我们装的钩子"，只增/改
    这一条，数组里其余用户自己的钩子（同一事件下的其它条目）原样保留。
    """
    events = _MANAGED_EVENTS + (_EXPERIMENTAL_EVENTS if include_experimental else ())

    merged = copy.deepcopy(existing)
    hooks_section = merged.setdefault("hooks", {})

    added: list[str] = []
    updated: list[str] = []
    for event_name in events:
        groups = hooks_section.setdefault(event_name, [])
        managed_hook = _find_managed_hook(groups, url)
        if managed_hook is None:
            groups.append(_build_hook_group(url, token))
            added.append(event_name)
        else:
            new_header = f"Bearer {token}"
            if managed_hook.get("headers", {}).get("Authorization") != new_header:
                managed_hook.setdefault("headers", {})["Authorization"] = new_header
                updated.append(event_name)

    return merged, InstallResult(added_events=tuple(added), updated_events=tuple(updated))


def install(
    settings_path: Path, *, url: str, token: str, include_experimental: bool = False
) -> InstallResult:
    """磁盘 I/O 外壳：读取（不存在视为空 dict）→ ``merge_hooks`` → 写临时文件 +
    ``os.replace`` 原子替换，防止 Claude Code 并发读到写一半的 ``settings.json``。

    调用方（``main.py``）负责用 try/except 包裹这次调用——``settings.json`` 格式异常
    时这里会直接抛出解析错误，绝不能让它中断 app 启动。
    """
    existing = (
        json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    )
    merged, result = merge_hooks(
        existing, url=url, token=token, include_experimental=include_experimental
    )

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = settings_path.parent / f"{settings_path.name}.tmp"
    tmp_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, settings_path)
    return result


def default_codex_hooks_path(*, project_dir: Path | None = None) -> Path:
    """``project_dir`` 给定 → ``<project_dir>/.codex/hooks.json``；否则 →
    ``~/.codex/hooks.json``（推断为与 ``config.toml`` 同目录，没有官方示例文件核实）。
    """
    base = project_dir if project_dir is not None else Path.home()
    return base / ".codex" / "hooks.json"


def default_gemini_settings_path(*, project_dir: Path | None = None) -> Path:
    """``project_dir`` 给定 → ``<project_dir>/.gemini/settings.json``；否则 →
    ``~/.gemini/settings.json``。
    """
    base = project_dir if project_dir is not None else Path.home()
    return base / ".gemini" / "settings.json"


def build_forward_command(*, url: str, token: str, source: str) -> str:
    """拼出调用 ``miku-on-desk-hook-forward``（见 ``face/hooks/forward.py``）的完整命令行，
    填进 Codex/Gemini 的 ``command`` 类型 hook。用 ``shlex.join`` 而不是手工拼接空格，
    避免 url/token 里出现空格或引号等特殊字符时命令行解析错位。
    """
    return shlex.join([_FORWARD_COMMAND_NAME, "--url", url, "--token", token, "--source", source])


def _build_command_hook_group(command: str) -> dict[str, Any]:
    return {"matcher": "", "hooks": [{"type": "command", "command": command}]}


def _find_managed_command_hook(groups: list[Any]) -> dict[str, Any] | None:
    """通过命令的第一个 token 是否等于 ``_FORWARD_COMMAND_NAME`` 识别"这是我们装的钩子"，
    而不要求整条命令字符串相等——url/token 会随 sidecar 每次启动变化，不能拿来做匹配 key。
    """
    for group in groups:
        for hook in group.get("hooks", []):
            if hook.get("type") != "command":
                continue
            existing_command = hook.get("command", "")
            try:
                first_token = shlex.split(existing_command)[0] if existing_command else ""
            except ValueError:
                continue
            if first_token == _FORWARD_COMMAND_NAME:
                return hook  # type: ignore[no-any-return]
    return None


def merge_codex_hooks(
    existing: dict[str, Any], *, url: str, token: str, include_experimental: bool
) -> tuple[dict[str, Any], InstallResult]:
    """纯函数，不做 I/O。``hooks.json`` 顶层直接以事件名做 key（不像 Claude Code 的
    ``settings.json`` 那样再包一层 ``"hooks"``——这一点没有官方示例文件核实,接入真实
    Codex 环境后需要重新确认）。
    """
    events = _MANAGED_EVENTS_CODEX + (_EXPERIMENTAL_EVENTS_CODEX if include_experimental else ())
    command = build_forward_command(url=url, token=token, source="codex")

    merged = copy.deepcopy(existing)

    added: list[str] = []
    updated: list[str] = []
    for event_name in events:
        groups = merged.setdefault(event_name, [])
        managed_hook = _find_managed_command_hook(groups)
        if managed_hook is None:
            groups.append(_build_command_hook_group(command))
            added.append(event_name)
        elif managed_hook.get("command") != command:
            managed_hook["command"] = command
            updated.append(event_name)

    return merged, InstallResult(added_events=tuple(added), updated_events=tuple(updated))


def install_codex(
    hooks_path: Path, *, url: str, token: str, include_experimental: bool = False
) -> InstallResult:
    """磁盘 I/O 外壳，逻辑与 ``install`` 对应，目标文件是 Codex CLI 的 ``hooks.json``。

    调用方负责用 try/except 包裹——与 ``install`` 一样，格式异常时这里直接抛出，绝不能
    让它中断 app 启动。
    """
    existing = json.loads(hooks_path.read_text(encoding="utf-8")) if hooks_path.exists() else {}
    merged, result = merge_codex_hooks(
        existing, url=url, token=token, include_experimental=include_experimental
    )

    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = hooks_path.parent / f"{hooks_path.name}.tmp"
    tmp_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, hooks_path)
    return result


def merge_gemini_hooks(
    existing: dict[str, Any], *, url: str, token: str, include_experimental: bool
) -> tuple[dict[str, Any], InstallResult]:
    """纯函数，不做 I/O。``settings.json`` 的 ``hooks`` 字段下按事件名分组，结构上与
    ``merge_hooks``（Claude Code）同构，只是 hook 类型是 ``command`` 而非 ``http``。
    """
    events = _MANAGED_EVENTS_GEMINI + (_EXPERIMENTAL_EVENTS_GEMINI if include_experimental else ())
    command = build_forward_command(url=url, token=token, source="gemini_cli")

    merged = copy.deepcopy(existing)
    hooks_section = merged.setdefault("hooks", {})

    added = []
    updated = []
    for event_name in events:
        groups = hooks_section.setdefault(event_name, [])
        managed_hook = _find_managed_command_hook(groups)
        if managed_hook is None:
            groups.append(_build_command_hook_group(command))
            added.append(event_name)
        elif managed_hook.get("command") != command:
            managed_hook["command"] = command
            updated.append(event_name)

    return merged, InstallResult(added_events=tuple(added), updated_events=tuple(updated))


def install_gemini(
    settings_path: Path, *, url: str, token: str, include_experimental: bool = False
) -> InstallResult:
    """磁盘 I/O 外壳，逻辑与 ``install`` 对应，目标文件是 Gemini CLI 的 ``settings.json``。"""
    existing = (
        json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    )
    merged, result = merge_gemini_hooks(
        existing, url=url, token=token, include_experimental=include_experimental
    )

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = settings_path.parent / f"{settings_path.name}.tmp"
    tmp_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, settings_path)
    return result
