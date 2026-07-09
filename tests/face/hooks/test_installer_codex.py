"""installer.py 里 Codex CLI 专属部分的回归测试：``merge_codex_hooks``（纯函数）与
``install_codex``（磁盘 I/O 外壳）。结构与 ``test_installer.py`` 对 Claude Code 的覆盖
方式对称，区别在于 hook 类型是 ``command`` 而非 ``http``，且没有外层 ``"hooks"`` 包裹。
"""

from __future__ import annotations

import json
from pathlib import Path

from miku_on_desk.face.hooks.installer import (
    _FORWARD_COMMAND_NAME,
    build_forward_command,
    default_codex_hooks_path,
    install_codex,
    merge_codex_hooks,
)

_URL = "http://127.0.0.1:8765/pet-event"
_MANAGED_EVENTS_CODEX = ("SessionStart", "UserPromptSubmit", "PostToolUse", "Stop")
_EXPERIMENTAL_EVENTS_CODEX = ("PreToolUse", "PermissionRequest")


def test_build_forward_command_starts_with_program_name_and_carries_args() -> None:
    command = build_forward_command(url=_URL, token="tok1", source="codex")

    assert command.startswith(_FORWARD_COMMAND_NAME)
    assert "--url" in command
    assert _URL in command
    assert "tok1" in command
    assert "codex" in command


def test_merge_codex_hooks_adds_all_managed_events_to_empty_config() -> None:
    merged, result = merge_codex_hooks({}, url=_URL, token="tok1", include_experimental=False)

    for event_name in _MANAGED_EVENTS_CODEX:
        groups = merged[event_name]
        assert len(groups) == 1
        hook = groups[0]["hooks"][0]
        assert hook["type"] == "command"
        assert hook["command"].startswith(_FORWARD_COMMAND_NAME)
    assert set(result.added_events) == set(_MANAGED_EVENTS_CODEX)
    assert result.updated_events == ()
    assert "hooks" not in merged


def test_merge_codex_hooks_excludes_experimental_events_by_default() -> None:
    merged, _ = merge_codex_hooks({}, url=_URL, token="tok1", include_experimental=False)

    for event_name in _EXPERIMENTAL_EVENTS_CODEX:
        assert event_name not in merged


def test_merge_codex_hooks_includes_experimental_events_when_opted_in() -> None:
    merged, result = merge_codex_hooks({}, url=_URL, token="tok1", include_experimental=True)

    for event_name in _EXPERIMENTAL_EVENTS_CODEX:
        assert event_name in merged
    assert set(result.added_events) == set(_MANAGED_EVENTS_CODEX) | set(_EXPERIMENTAL_EVENTS_CODEX)


def test_merge_codex_hooks_rerun_with_same_token_is_idempotent() -> None:
    first, _ = merge_codex_hooks({}, url=_URL, token="tok1", include_experimental=False)

    second, result = merge_codex_hooks(first, url=_URL, token="tok1", include_experimental=False)

    assert second == first
    assert result.added_events == ()
    assert result.updated_events == ()
    for event_name in _MANAGED_EVENTS_CODEX:
        assert len(second[event_name]) == 1


def test_merge_codex_hooks_rerun_with_new_token_refreshes_command_only() -> None:
    first, _ = merge_codex_hooks({}, url=_URL, token="tok1", include_experimental=False)

    second, result = merge_codex_hooks(first, url=_URL, token="tok2", include_experimental=False)

    assert set(result.updated_events) == set(_MANAGED_EVENTS_CODEX)
    assert result.added_events == ()
    for event_name in _MANAGED_EVENTS_CODEX:
        groups = second[event_name]
        assert len(groups) == 1
        assert "tok2" in groups[0]["hooks"][0]["command"]


def test_merge_codex_hooks_preserves_unrelated_existing_hooks_untouched() -> None:
    existing = {
        "SessionStart": [{"matcher": "", "hooks": [{"type": "command", "command": "echo hi"}]}],
        "SomeOtherEvent": [{"matcher": "", "hooks": [{"type": "command", "command": "x"}]}],
    }

    merged, result = merge_codex_hooks(existing, url=_URL, token="tok1", include_experimental=False)

    session_start_groups = merged["SessionStart"]
    assert len(session_start_groups) == 2
    assert session_start_groups[0]["hooks"][0]["command"] == "echo hi"
    assert merged["SomeOtherEvent"] == existing["SomeOtherEvent"]
    assert "SessionStart" in result.added_events


def test_merge_codex_hooks_does_not_mutate_input() -> None:
    existing: dict[str, object] = {}

    merge_codex_hooks(existing, url=_URL, token="tok1", include_experimental=False)

    assert existing == {}


def test_default_codex_hooks_path_without_project_dir_uses_home() -> None:
    path = default_codex_hooks_path()

    assert path == Path.home() / ".codex" / "hooks.json"


def test_default_codex_hooks_path_with_project_dir(tmp_path: Path) -> None:
    path = default_codex_hooks_path(project_dir=tmp_path)

    assert path == tmp_path / ".codex" / "hooks.json"


def test_install_codex_creates_hooks_file_when_missing(tmp_path: Path) -> None:
    hooks_path = tmp_path / ".codex" / "hooks.json"

    result = install_codex(hooks_path, url=_URL, token="tok1")

    assert hooks_path.exists()
    written = json.loads(hooks_path.read_text(encoding="utf-8"))
    assert written["SessionStart"][0]["hooks"][0]["command"].startswith(_FORWARD_COMMAND_NAME)
    assert set(result.added_events) == set(_MANAGED_EVENTS_CODEX)


def test_install_codex_rerun_updates_token_without_duplicating(tmp_path: Path) -> None:
    hooks_path = tmp_path / ".codex" / "hooks.json"
    install_codex(hooks_path, url=_URL, token="tok1")

    result = install_codex(hooks_path, url=_URL, token="tok2")

    written = json.loads(hooks_path.read_text(encoding="utf-8"))
    groups = written["SessionStart"]
    assert len(groups) == 1
    assert "tok2" in groups[0]["hooks"][0]["command"]
    assert set(result.updated_events) == set(_MANAGED_EVENTS_CODEX)


def test_install_codex_preserves_existing_unrelated_settings(tmp_path: Path) -> None:
    hooks_path = tmp_path / ".codex" / "hooks.json"
    hooks_path.parent.mkdir(parents=True)
    hooks_path.write_text(json.dumps({"SomeOtherEvent": [1, 2, 3]}), encoding="utf-8")

    install_codex(hooks_path, url=_URL, token="tok1")

    written = json.loads(hooks_path.read_text(encoding="utf-8"))
    assert written["SomeOtherEvent"] == [1, 2, 3]
