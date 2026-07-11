"""installer.py 里 Gemini CLI 专属部分的回归测试：``merge_gemini_hooks``（纯函数）与
``install_gemini``（磁盘 I/O 外壳）。结构上与 Claude Code 的 ``merge_hooks`` 同构（同样
在 ``"hooks"`` 字段下按事件名分组），区别只是 hook 类型是 ``command`` 而非 ``http``。
"""

from __future__ import annotations

import json
from pathlib import Path

from miku_on_desk.face.hooks.installer import (
    _FORWARD_COMMAND_NAME,
    default_gemini_settings_path,
    install_gemini,
    merge_gemini_hooks,
)

_URL = "http://127.0.0.1:8765/pet-event"
_MANAGED_EVENTS_GEMINI = ("SessionStart", "SessionEnd", "AfterTool", "AfterAgent")
_EXPERIMENTAL_EVENTS_GEMINI = ("BeforeTool", "BeforeAgent", "Notification")


def test_merge_gemini_hooks_adds_all_managed_events_to_empty_config() -> None:
    merged, result = merge_gemini_hooks({}, url=_URL, token="tok1", include_experimental=False)

    for event_name in _MANAGED_EVENTS_GEMINI:
        groups = merged["hooks"][event_name]
        assert len(groups) == 1
        hook = groups[0]["hooks"][0]
        assert hook["type"] == "command"
        assert hook["command"].startswith(_FORWARD_COMMAND_NAME)
    assert set(result.added_events) == set(_MANAGED_EVENTS_GEMINI)
    assert result.updated_events == ()


def test_merge_gemini_hooks_excludes_experimental_events_by_default() -> None:
    merged, _ = merge_gemini_hooks({}, url=_URL, token="tok1", include_experimental=False)

    for event_name in _EXPERIMENTAL_EVENTS_GEMINI:
        assert event_name not in merged.get("hooks", {})


def test_merge_gemini_hooks_includes_experimental_events_when_opted_in() -> None:
    merged, result = merge_gemini_hooks({}, url=_URL, token="tok1", include_experimental=True)

    for event_name in _EXPERIMENTAL_EVENTS_GEMINI:
        assert event_name in merged["hooks"]
    assert set(result.added_events) == set(_MANAGED_EVENTS_GEMINI) | set(
        _EXPERIMENTAL_EVENTS_GEMINI
    )


def test_merge_gemini_hooks_rerun_with_same_token_is_idempotent() -> None:
    first, _ = merge_gemini_hooks({}, url=_URL, token="tok1", include_experimental=False)

    second, result = merge_gemini_hooks(first, url=_URL, token="tok1", include_experimental=False)

    assert second == first
    assert result.added_events == ()
    assert result.updated_events == ()


def test_merge_gemini_hooks_rerun_with_new_token_refreshes_command_only() -> None:
    first, _ = merge_gemini_hooks({}, url=_URL, token="tok1", include_experimental=False)

    second, result = merge_gemini_hooks(first, url=_URL, token="tok2", include_experimental=False)

    assert set(result.updated_events) == set(_MANAGED_EVENTS_GEMINI)
    for event_name in _MANAGED_EVENTS_GEMINI:
        groups = second["hooks"][event_name]
        assert len(groups) == 1
        assert "tok2" in groups[0]["hooks"][0]["command"]


def test_merge_gemini_hooks_preserves_unrelated_existing_hooks_untouched() -> None:
    existing = {
        "hooks": {
            "SessionStart": [{"matcher": "", "hooks": [{"type": "command", "command": "echo hi"}]}],
            "SomeOtherEvent": [{"matcher": "", "hooks": [{"type": "command", "command": "x"}]}],
        }
    }

    merged, result = merge_gemini_hooks(
        existing, url=_URL, token="tok1", include_experimental=False
    )

    session_start_groups = merged["hooks"]["SessionStart"]
    assert len(session_start_groups) == 2
    assert merged["hooks"]["SomeOtherEvent"] == existing["hooks"]["SomeOtherEvent"]
    assert "SessionStart" in result.added_events


def test_merge_gemini_hooks_does_not_mutate_input() -> None:
    existing: dict[str, object] = {}

    merge_gemini_hooks(existing, url=_URL, token="tok1", include_experimental=False)

    assert existing == {}


def test_default_gemini_settings_path_without_project_dir_uses_home() -> None:
    path = default_gemini_settings_path()

    assert path == Path.home() / ".gemini" / "settings.json"


def test_default_gemini_settings_path_with_project_dir(tmp_path: Path) -> None:
    path = default_gemini_settings_path(project_dir=tmp_path)

    assert path == tmp_path / ".gemini" / "settings.json"


def test_install_gemini_creates_settings_file_when_missing(tmp_path: Path) -> None:
    settings_path = tmp_path / ".gemini" / "settings.json"

    result = install_gemini(settings_path, url=_URL, token="tok1")

    assert settings_path.exists()
    written = json.loads(settings_path.read_text(encoding="utf-8"))
    assert written["hooks"]["SessionStart"][0]["hooks"][0]["command"].startswith(
        _FORWARD_COMMAND_NAME
    )
    assert set(result.added_events) == set(_MANAGED_EVENTS_GEMINI)


def test_install_gemini_rerun_updates_token_without_duplicating(tmp_path: Path) -> None:
    settings_path = tmp_path / ".gemini" / "settings.json"
    install_gemini(settings_path, url=_URL, token="tok1")

    result = install_gemini(settings_path, url=_URL, token="tok2")

    written = json.loads(settings_path.read_text(encoding="utf-8"))
    groups = written["hooks"]["SessionStart"]
    assert len(groups) == 1
    assert "tok2" in groups[0]["hooks"][0]["command"]
    assert set(result.updated_events) == set(_MANAGED_EVENTS_GEMINI)


def test_install_gemini_preserves_existing_unrelated_settings(tmp_path: Path) -> None:
    settings_path = tmp_path / ".gemini" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(json.dumps({"some_other_setting": True}), encoding="utf-8")

    install_gemini(settings_path, url=_URL, token="tok1")

    written = json.loads(settings_path.read_text(encoding="utf-8"))
    assert written["some_other_setting"] is True
