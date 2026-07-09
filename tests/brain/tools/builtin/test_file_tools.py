"""file_tools.py 的回归测试：read_file/write_file 的沙箱越权、先读后写、截断、原子写等路径。"""

from __future__ import annotations

from pathlib import Path

from miku_on_desk.brain.providers.base import ToolUseBlock
from miku_on_desk.brain.tools.builtin import file_tools as file_tools_module
from miku_on_desk.brain.tools.builtin.file_tools import register_file_tools
from miku_on_desk.brain.tools.path_sandbox import PathSandbox
from miku_on_desk.brain.tools.policy import Decision, PolicyEngine
from miku_on_desk.brain.tools.read_tracker import ReadTracker
from miku_on_desk.brain.tools.registry import ToolRegistry


def _make_registry(tmp_path: Path, *, trusted_mode: bool = True) -> tuple[ToolRegistry, Path]:
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    sandbox = PathSandbox(cwd=cwd, output_dir=tmp_path / "output", data_dir=tmp_path / "data")
    read_tracker = ReadTracker()
    policy = PolicyEngine(
        trusted_mode=trusted_mode,
        allowed_tools=frozenset(),
        denied_tools=frozenset(),
        default_decision=Decision.ALLOW,
        path_sandbox=sandbox,
        read_tracker=read_tracker,
    )
    registry = ToolRegistry(policy, read_tracker)
    register_file_tools(registry)
    return registry, cwd


async def test_read_file_returns_content(tmp_path: Path) -> None:
    registry, cwd = _make_registry(tmp_path)
    target = cwd / "hello.txt"
    target.write_text("hello world", encoding="utf-8")

    result = await registry.execute(
        ToolUseBlock(id="c1", name="read_file", input={"path": str(target)}), session_id="s1"
    )

    assert result.is_error is False
    assert result.content == "hello world"


async def test_read_file_missing_path_returns_friendly_message_and_marks_read(
    tmp_path: Path,
) -> None:
    registry, cwd = _make_registry(tmp_path, trusted_mode=False)
    target = cwd / "new_file.txt"

    result = await registry.execute(
        ToolUseBlock(id="c1", name="read_file", input={"path": str(target)}), session_id="s1"
    )

    assert result.is_error is False
    assert "不存在" in result.content
    assert "write_file" in result.content

    decision = registry.evaluate(
        ToolUseBlock(id="c2", name="write_file", input={"path": str(target), "content": "new"}),
        session_id="s1",
    )
    assert decision.decision is not Decision.DENY


async def test_read_file_directory_is_error(tmp_path: Path) -> None:
    registry, cwd = _make_registry(tmp_path)

    result = await registry.execute(
        ToolUseBlock(id="c1", name="read_file", input={"path": str(cwd)}), session_id="s1"
    )

    assert result.is_error is True
    assert "exec_command" in result.content


async def test_read_file_truncates_large_content(tmp_path: Path, monkeypatch: object) -> None:
    registry, cwd = _make_registry(tmp_path)
    target = cwd / "big.txt"
    target.write_text("x" * 100, encoding="utf-8")
    monkeypatch.setattr(file_tools_module, "_MAX_READ_FILE_CHARS", 10)  # type: ignore[attr-defined]

    result = await registry.execute(
        ToolUseBlock(id="c1", name="read_file", input={"path": str(target)}), session_id="s1"
    )

    assert result.is_error is False
    assert result.content.startswith("x" * 10)
    assert "已截断" in result.content


async def test_write_file_denied_without_prior_read(tmp_path: Path) -> None:
    registry, cwd = _make_registry(tmp_path)
    target = cwd / "out.txt"

    decision = registry.evaluate(
        ToolUseBlock(id="c1", name="write_file", input={"path": str(target), "content": "hi"}),
        session_id="s1",
    )

    assert decision.decision is Decision.DENY
    assert "read_file" in (decision.reason or "")


async def test_write_file_outside_sandbox_denied(tmp_path: Path) -> None:
    registry, _cwd = _make_registry(tmp_path)
    outside = tmp_path.parent / "definitely-outside-sandbox.txt"

    decision = registry.evaluate(
        ToolUseBlock(id="c1", name="write_file", input={"path": str(outside), "content": "hi"}),
        session_id="s1",
    )

    assert decision.decision is Decision.DENY


async def test_write_file_allowed_after_read_file_and_writes_atomically(
    tmp_path: Path,
) -> None:
    registry, cwd = _make_registry(tmp_path, trusted_mode=True)
    target = cwd / "nested" / "out.txt"

    await registry.execute(
        ToolUseBlock(id="c1", name="read_file", input={"path": str(target)}), session_id="s1"
    )

    decision = registry.evaluate(
        ToolUseBlock(id="c2", name="write_file", input={"path": str(target), "content": "hi"}),
        session_id="s1",
    )
    assert decision.decision is Decision.ALLOW

    result = await registry.execute(
        ToolUseBlock(id="c2", name="write_file", input={"path": str(target), "content": "hi"}),
        session_id="s1",
    )

    assert result.is_error is False
    assert target.read_text(encoding="utf-8") == "hi"
    assert list(target.parent.glob("*.tmp-*")) == []


async def test_evaluate_requires_confirmation_for_write_when_not_trusted(
    tmp_path: Path,
) -> None:
    registry, cwd = _make_registry(tmp_path, trusted_mode=False)
    target = cwd / "out.txt"
    target.write_text("existing", encoding="utf-8")

    await registry.execute(
        ToolUseBlock(id="c1", name="read_file", input={"path": str(target)}), session_id="s1"
    )

    decision = registry.evaluate(
        ToolUseBlock(id="c2", name="write_file", input={"path": str(target), "content": "hi"}),
        session_id="s1",
    )

    assert decision.decision is Decision.ASK
