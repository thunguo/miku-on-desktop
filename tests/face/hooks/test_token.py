"""token.py 的回归测试：权限位、每次轮换生成不同值、写入内容可读回。"""

from __future__ import annotations

import stat
from pathlib import Path

from miku_on_desk.face.hooks.token import generate_token, rotate_token, write_token


def test_generate_token_produces_hex_string_of_expected_length() -> None:
    token = generate_token()

    assert len(token) == 64
    int(token, 16)  # 应能被解析为十六进制，否则抛 ValueError


def test_generate_token_is_random_across_calls() -> None:
    assert generate_token() != generate_token()


def test_write_token_creates_file_with_owner_only_permissions(tmp_path: Path) -> None:
    token_path = tmp_path / "hook_token"

    write_token(token_path, "abc123")

    mode = stat.S_IMODE(token_path.stat().st_mode)
    assert mode == 0o600


def test_write_token_content_is_readable_back(tmp_path: Path) -> None:
    token_path = tmp_path / "hook_token"

    write_token(token_path, "the-token-value")

    assert token_path.read_text(encoding="utf-8") == "the-token-value"


def test_write_token_creates_parent_directories(tmp_path: Path) -> None:
    token_path = tmp_path / "nested" / "dir" / "hook_token"

    write_token(token_path, "v")

    assert token_path.exists()


def test_write_token_overwrites_existing_file(tmp_path: Path) -> None:
    token_path = tmp_path / "hook_token"
    write_token(token_path, "old")

    write_token(token_path, "new")

    assert token_path.read_text(encoding="utf-8") == "new"


def test_rotate_token_returns_newly_written_value(tmp_path: Path) -> None:
    token_path = tmp_path / "hook_token"

    token = rotate_token(token_path)

    assert token_path.read_text(encoding="utf-8") == token


def test_rotate_token_produces_different_value_each_call(tmp_path: Path) -> None:
    token_path = tmp_path / "hook_token"

    first = rotate_token(token_path)
    second = rotate_token(token_path)

    assert first != second
    assert token_path.read_text(encoding="utf-8") == second
