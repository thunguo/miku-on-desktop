"""SecretVault 的加密存储回归测试：确保落盘密文、密钥文件权限、CRUD 接口均按预期工作。"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from miku_on_desk.brain.secrets.vault import SecretVault


@pytest.fixture
def vault(tmp_path: Path) -> SecretVault:
    return SecretVault(db_path=tmp_path / "secrets.db", key_path=tmp_path / "secrets.key")


def test_store_and_get_roundtrip(vault: SecretVault) -> None:
    vault.store("anthropic_api_key", "sk-ant-super-secret")
    assert vault.get("anthropic_api_key") == "sk-ant-super-secret"


def test_get_missing_key_returns_none(vault: SecretVault) -> None:
    assert vault.get("does-not-exist") is None


def test_store_overwrites_existing_value(vault: SecretVault) -> None:
    vault.store("k", "v1")
    vault.store("k", "v2")
    assert vault.get("k") == "v2"
    assert vault.list_keys() == ["k"]


def test_delete_removes_key(vault: SecretVault) -> None:
    vault.store("k", "v")
    vault.delete("k")
    assert vault.get("k") is None
    assert vault.list_keys() == []


def test_list_keys_sorted(vault: SecretVault) -> None:
    vault.store("b", "1")
    vault.store("a", "2")
    assert vault.list_keys() == ["a", "b"]


def test_value_is_never_stored_as_plaintext_on_disk(tmp_path: Path) -> None:
    db_path = tmp_path / "secrets.db"
    vault = SecretVault(db_path=db_path, key_path=tmp_path / "secrets.key")
    vault.store("k", "super-secret-plaintext-marker")
    vault.close()

    assert b"super-secret-plaintext-marker" not in db_path.read_bytes()


def test_master_key_file_has_owner_only_permissions(tmp_path: Path) -> None:
    key_path = tmp_path / "secrets.key"
    SecretVault(db_path=tmp_path / "secrets.db", key_path=key_path)

    mode = stat.S_IMODE(key_path.stat().st_mode)
    assert mode == 0o600


def test_master_key_is_reused_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "secrets.db"
    key_path = tmp_path / "secrets.key"

    first = SecretVault(db_path=db_path, key_path=key_path)
    first.store("k", "v")
    first.close()

    second = SecretVault(db_path=db_path, key_path=key_path)
    assert second.get("k") == "v"
