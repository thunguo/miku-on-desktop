"""installer.py 的回归测试：``merge_hooks`` 的幂等合并逻辑（纯函数）与 ``install`` 的
磁盘 I/O 外壳。
"""

from __future__ import annotations

import json
from pathlib import Path

from miku_on_desk.face.hooks.installer import (
    default_claude_settings_path,
    install,
    merge_hooks,
)

_URL = "http://127.0.0.1:8765/pet-event"
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


def test_merge_hooks_adds_all_managed_events_to_empty_config() -> None:
    merged, result = merge_hooks({}, url=_URL, token="tok1", include_experimental=False)

    for event_name in _MANAGED_EVENTS:
        groups = merged["hooks"][event_name]
        assert len(groups) == 1
        hook = groups[0]["hooks"][0]
        assert hook["type"] == "http"
        assert hook["url"] == _URL
        assert hook["headers"]["Authorization"] == "Bearer tok1"
    assert set(result.added_events) == set(_MANAGED_EVENTS)
    assert result.updated_events == ()


def test_merge_hooks_excludes_experimental_events_by_default() -> None:
    merged, _ = merge_hooks({}, url=_URL, token="tok1", include_experimental=False)

    for event_name in _EXPERIMENTAL_EVENTS:
        assert event_name not in merged.get("hooks", {})


def test_merge_hooks_includes_experimental_events_when_opted_in() -> None:
    merged, result = merge_hooks({}, url=_URL, token="tok1", include_experimental=True)

    for event_name in _EXPERIMENTAL_EVENTS:
        assert event_name in merged["hooks"]
    assert set(result.added_events) == set(_MANAGED_EVENTS) | set(_EXPERIMENTAL_EVENTS)


def test_merge_hooks_rerun_with_same_token_is_idempotent() -> None:
    first, _ = merge_hooks({}, url=_URL, token="tok1", include_experimental=False)

    second, result = merge_hooks(first, url=_URL, token="tok1", include_experimental=False)

    assert second == first
    assert result.added_events == ()
    assert result.updated_events == ()
    for event_name in _MANAGED_EVENTS:
        assert len(second["hooks"][event_name]) == 1


def test_merge_hooks_rerun_with_new_token_refreshes_header_only() -> None:
    first, _ = merge_hooks({}, url=_URL, token="tok1", include_experimental=False)

    second, result = merge_hooks(first, url=_URL, token="tok2", include_experimental=False)

    assert set(result.updated_events) == set(_MANAGED_EVENTS)
    assert result.added_events == ()
    for event_name in _MANAGED_EVENTS:
        groups = second["hooks"][event_name]
        assert len(groups) == 1
        assert groups[0]["hooks"][0]["headers"]["Authorization"] == "Bearer tok2"


def test_merge_hooks_preserves_unrelated_existing_hooks_untouched() -> None:
    existing = {
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "",
                    "hooks": [{"type": "command", "command": "echo hi"}],
                }
            ],
            "SomeOtherEvent": [{"matcher": "", "hooks": [{"type": "command", "command": "x"}]}],
        }
    }

    merged, result = merge_hooks(existing, url=_URL, token="tok1", include_experimental=False)

    session_start_groups = merged["hooks"]["SessionStart"]
    assert len(session_start_groups) == 2
    assert session_start_groups[0]["hooks"][0]["type"] == "command"
    assert merged["hooks"]["SomeOtherEvent"] == existing["hooks"]["SomeOtherEvent"]
    assert "SessionStart" in result.added_events


def test_merge_hooks_does_not_mutate_input() -> None:
    existing: dict[str, object] = {}

    merge_hooks(existing, url=_URL, token="tok1", include_experimental=False)

    assert existing == {}


def test_default_claude_settings_path_without_project_dir_uses_home() -> None:
    path = default_claude_settings_path()

    assert path == Path.home() / ".claude" / "settings.json"


def test_default_claude_settings_path_with_project_dir(tmp_path: Path) -> None:
    path = default_claude_settings_path(project_dir=tmp_path)

    assert path == tmp_path / ".claude" / "settings.json"


def test_install_creates_settings_file_when_missing(tmp_path: Path) -> None:
    settings_path = tmp_path / ".claude" / "settings.json"

    result = install(settings_path, url=_URL, token="tok1")

    assert settings_path.exists()
    written = json.loads(settings_path.read_text(encoding="utf-8"))
    assert written["hooks"]["SessionStart"][0]["hooks"][0]["url"] == _URL
    assert set(result.added_events) == set(_MANAGED_EVENTS)


def test_install_rerun_updates_token_without_duplicating(tmp_path: Path) -> None:
    settings_path = tmp_path / ".claude" / "settings.json"
    install(settings_path, url=_URL, token="tok1")

    result = install(settings_path, url=_URL, token="tok2")

    written = json.loads(settings_path.read_text(encoding="utf-8"))
    groups = written["hooks"]["SessionStart"]
    assert len(groups) == 1
    assert groups[0]["hooks"][0]["headers"]["Authorization"] == "Bearer tok2"
    assert set(result.updated_events) == set(_MANAGED_EVENTS)


def test_install_preserves_existing_unrelated_settings(tmp_path: Path) -> None:
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(json.dumps({"some_other_setting": True}), encoding="utf-8")

    install(settings_path, url=_URL, token="tok1")

    written = json.loads(settings_path.read_text(encoding="utf-8"))
    assert written["some_other_setting"] is True
