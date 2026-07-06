"""把本地 hook sidecar 接入 Claude Code 的 ``settings.json``。

``http`` 类型的 hook 是 Claude Code 官方支持的配置方式：会同步 POST 事件 JSON 到给定
URL 并等待响应（默认超时 600s），不需要用户自己写 shell 脚本转发。

只接入纯通知性质的事件（``_MANAGED_EVENTS``）：``PreToolUse``/``PermissionRequest``/
``PermissionDenied`` 的 HTTP 响应体可能被 Claude Code 当作真正的允许/拒绝决策使用
（这点在公开文档里未被百分百确认），而本项目的 sidecar 只是想要视觉反馈、不追求协议
对接，所以这三个事件放进 ``_EXPERIMENTAL_EVENTS``，需要显式 opt-in（
``include_experimental=True``），且启用前必须对照当时最新的官方文档重新确认响应体
语义，否则错误的响应可能意外拦下真实的工具调用。

Codex CLI / Gemini CLI 的适配是尚未实现的预留扩展点，不对它们的配置格式做没有依据的
猜测实现。
"""

from __future__ import annotations

import copy
import json
import os
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
