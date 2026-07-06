"""PathSandbox 的单元测试：验证允许根目录的拼装规则与拒绝时的引导文案。"""

from __future__ import annotations

from pathlib import Path

import pytest

from miku_on_desk.brain.tools.path_sandbox import PathSandbox


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PathSandbox:
    # pytest 的 tmp_path 本身位于系统临时目录之下，而 PathSandbox 总是把
    # tempfile.gettempdir() 纳入允许根目录——不隔离的话，任何 "outside" 测试路径
    # 都会被临时目录这一条规则误判为允许。
    monkeypatch.setattr(
        "miku_on_desk.brain.tools.path_sandbox.tempfile.gettempdir",
        lambda: str(tmp_path / "system_tmp"),
    )
    cwd = tmp_path / "cwd"
    output_dir = tmp_path / "output"
    data_dir = tmp_path / "data"
    for d in (cwd, output_dir, data_dir):
        d.mkdir()
    return PathSandbox(cwd=cwd, output_dir=output_dir, data_dir=data_dir)


def test_allows_exact_root(sandbox: PathSandbox, tmp_path: Path) -> None:
    result = sandbox.check(tmp_path / "cwd")
    assert result.allowed is True


def test_allows_nested_path_under_root(sandbox: PathSandbox, tmp_path: Path) -> None:
    result = sandbox.check(tmp_path / "cwd" / "sub" / "file.txt")
    assert result.allowed is True


def test_allows_path_under_output_dir(sandbox: PathSandbox, tmp_path: Path) -> None:
    result = sandbox.check(tmp_path / "output" / "artifact.png")
    assert result.allowed is True


def test_denies_path_outside_all_roots(sandbox: PathSandbox, tmp_path: Path) -> None:
    result = sandbox.check(tmp_path / "outside" / "secret.txt")
    assert result.allowed is False
    assert result.reason is not None
    assert "不要用同一路径重试" in result.reason


def test_home_subdir_included_when_it_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "miku_on_desk.brain.tools.path_sandbox.tempfile.gettempdir",
        lambda: str(tmp_path / "system_tmp"),
    )
    fake_home = tmp_path / "home"
    (fake_home / "code").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    cwd = tmp_path / "cwd"
    output_dir = tmp_path / "output"
    data_dir = tmp_path / "data"
    for d in (cwd, output_dir, data_dir):
        d.mkdir()
    sandbox = PathSandbox(cwd=cwd, output_dir=output_dir, data_dir=data_dir)

    result = sandbox.check(fake_home / "code" / "project" / "main.py")
    assert result.allowed is True


def test_home_subdir_excluded_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "miku_on_desk.brain.tools.path_sandbox.tempfile.gettempdir",
        lambda: str(tmp_path / "system_tmp"),
    )
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    cwd = tmp_path / "cwd"
    output_dir = tmp_path / "output"
    data_dir = tmp_path / "data"
    for d in (cwd, output_dir, data_dir):
        d.mkdir()
    sandbox = PathSandbox(cwd=cwd, output_dir=output_dir, data_dir=data_dir)

    result = sandbox.check(fake_home / "code" / "project" / "main.py")
    assert result.allowed is False


def test_extra_dirs_are_allowed(tmp_path: Path) -> None:
    cwd = tmp_path / "cwd"
    output_dir = tmp_path / "output"
    data_dir = tmp_path / "data"
    extra = tmp_path / "extra"
    for d in (cwd, output_dir, data_dir, extra):
        d.mkdir()
    sandbox = PathSandbox(cwd=cwd, output_dir=output_dir, data_dir=data_dir, extra_dirs=[extra])

    result = sandbox.check(extra / "file.txt")
    assert result.allowed is True
