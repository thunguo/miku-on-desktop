"""PolicyEngine 的单元测试：验证七层闸门的判断顺序与豁免边界。

重点覆盖 VL-023（requires_confirmation 不受会话授权豁免）与"信任层不能豁免路径沙箱/先读后改"
这两条本模块最容易被破坏的不变量。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from miku_on_desk.brain.tools.path_sandbox import PathSandbox
from miku_on_desk.brain.tools.policy import Decision, PolicyEngine, ToolPolicySpec
from miku_on_desk.brain.tools.read_tracker import ReadTracker

_SESSION = "session-1"


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PathSandbox:
    # 与 test_path_sandbox.py 中的理由相同：隔离系统临时目录，避免 tmp_path
    # 落在其下导致 "outside" 类测试路径被临时目录规则误判为允许。
    monkeypatch.setattr(
        "miku_on_desk.brain.tools.path_sandbox.tempfile.gettempdir",
        lambda: str(tmp_path / "system_tmp"),
    )
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    return PathSandbox(cwd=cwd, output_dir=tmp_path / "output", data_dir=tmp_path / "data")


def _engine(sandbox: PathSandbox, **overrides: object) -> PolicyEngine:
    defaults: dict[str, object] = {
        "trusted_mode": False,
        "allowed_tools": frozenset(),
        "denied_tools": frozenset(),
        "default_decision": Decision.ASK,
        "path_sandbox": sandbox,
        "read_tracker": ReadTracker(),
    }
    defaults.update(overrides)
    return PolicyEngine(**defaults)  # type: ignore[arg-type]


def test_denied_tools_wins_over_everything(sandbox: PathSandbox) -> None:
    engine = _engine(sandbox, trusted_mode=True, denied_tools=frozenset({"do_thing"}))
    decision = engine.evaluate("do_thing", {}, ToolPolicySpec(), session_id=_SESSION)
    assert decision.decision == Decision.DENY


def test_default_decision_is_returned_when_nothing_else_matches(sandbox: PathSandbox) -> None:
    engine = _engine(sandbox, default_decision=Decision.ASK)
    decision = engine.evaluate("do_thing", {}, ToolPolicySpec(), session_id=_SESSION)
    assert decision.decision == Decision.ASK


def test_path_outside_sandbox_denies_even_when_trusted(
    sandbox: PathSandbox, tmp_path: Path
) -> None:
    engine = _engine(sandbox, trusted_mode=True)
    spec = ToolPolicySpec(path_arg="path")
    outside = str(tmp_path / "outside" / "secret.txt")
    decision = engine.evaluate("read_file", {"path": outside}, spec, session_id=_SESSION)
    assert decision.decision == Decision.DENY
    assert decision.reason is not None and "不要用同一路径重试" in decision.reason


def test_missing_path_arg_denies(sandbox: PathSandbox) -> None:
    engine = _engine(sandbox)
    spec = ToolPolicySpec(path_arg="path")
    decision = engine.evaluate("read_file", {}, spec, session_id=_SESSION)
    assert decision.decision == Decision.DENY


def test_write_without_prior_read_denies_even_when_trusted(
    sandbox: PathSandbox, tmp_path: Path
) -> None:
    engine = _engine(sandbox, trusted_mode=True)
    spec = ToolPolicySpec(path_arg="path", is_write=True)
    target = str(tmp_path / "cwd" / "a.txt")
    decision = engine.evaluate("write_file", {"path": target}, spec, session_id=_SESSION)
    assert decision.decision == Decision.DENY
    assert decision.reason is not None and "先用 read_file 读取过" in decision.reason


def test_write_after_prior_read_is_allowed_when_trusted(
    sandbox: PathSandbox, tmp_path: Path
) -> None:
    read_tracker = ReadTracker()
    target = tmp_path / "cwd" / "a.txt"
    read_tracker.mark_read(_SESSION, target)
    engine = _engine(sandbox, trusted_mode=True, read_tracker=read_tracker)
    spec = ToolPolicySpec(path_arg="path", is_write=True)
    decision = engine.evaluate("write_file", {"path": str(target)}, spec, session_id=_SESSION)
    assert decision.decision == Decision.ALLOW


def test_requires_confirmation_asks_even_when_tool_is_allow_listed(sandbox: PathSandbox) -> None:
    engine = _engine(sandbox, allowed_tools=frozenset({"do_thing"}))
    spec = ToolPolicySpec(requires_confirmation=True, confirm_reason="危险操作")
    decision = engine.evaluate("do_thing", {}, spec, session_id=_SESSION)
    assert decision.decision == Decision.ASK
    assert decision.reason == "危险操作"


def test_requires_confirmation_asks_even_when_previously_granted(sandbox: PathSandbox) -> None:
    engine = _engine(sandbox)
    engine.grant(_SESSION, "do_thing")
    spec = ToolPolicySpec(requires_confirmation=True)
    decision = engine.evaluate("do_thing", {}, spec, session_id=_SESSION)
    assert decision.decision == Decision.ASK


def test_trusted_mode_bypasses_requires_confirmation(sandbox: PathSandbox) -> None:
    engine = _engine(sandbox, trusted_mode=True)
    spec = ToolPolicySpec(requires_confirmation=True)
    decision = engine.evaluate("do_thing", {}, spec, session_id=_SESSION)
    assert decision.decision == Decision.ALLOW


def test_allow_listed_tool_is_allowed(sandbox: PathSandbox) -> None:
    engine = _engine(sandbox, allowed_tools=frozenset({"do_thing"}))
    decision = engine.evaluate("do_thing", {}, ToolPolicySpec(), session_id=_SESSION)
    assert decision.decision == Decision.ALLOW


def test_session_grant_is_allowed(sandbox: PathSandbox) -> None:
    engine = _engine(sandbox)
    engine.grant(_SESSION, "do_thing")
    decision = engine.evaluate("do_thing", {}, ToolPolicySpec(), session_id=_SESSION)
    assert decision.decision == Decision.ALLOW


def test_session_grant_does_not_leak_across_sessions(sandbox: PathSandbox) -> None:
    engine = _engine(sandbox)
    engine.grant(_SESSION, "do_thing")
    decision = engine.evaluate("do_thing", {}, ToolPolicySpec(), session_id="other-session")
    assert decision.decision != Decision.ALLOW


@pytest.mark.parametrize("command", ["a; b", "a && b || c", "cat a | grep b"])
def test_dangerous_command_regex_matches_single_metacharacters_not_double(
    sandbox: PathSandbox, command: str
) -> None:
    engine = _engine(sandbox)
    spec = ToolPolicySpec(command_arg="command")
    decision = engine.evaluate("exec_command", {"command": command}, spec, session_id=_SESSION)
    if command == "a && b || c":
        assert decision.decision == Decision.ASK  # falls through to default (also "ask")
        assert decision.reason is None
    else:
        assert decision.decision == Decision.ASK
        assert decision.reason is not None and "需要人工确认的字符" in decision.reason


def test_clear_session_removes_grant(sandbox: PathSandbox) -> None:
    engine = _engine(sandbox)
    engine.grant(_SESSION, "do_thing")
    engine.clear_session(_SESSION)
    decision = engine.evaluate("do_thing", {}, ToolPolicySpec(), session_id=_SESSION)
    assert decision.decision != Decision.ALLOW
