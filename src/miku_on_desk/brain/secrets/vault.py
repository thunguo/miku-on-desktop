"""本机密钥保管：AES-256-GCM 加密存储，主密钥为随机字节，以 0600 权限的十六进制文件持久化。

主密钥不做密码派生（不是 KDF 出来的），只依赖操作系统文件权限保护——威胁模型是防止密钥以
明文形式出现在磁盘快照/备份/误传日志里，不是防御已登录本机账户本身的攻击者（单机单用户场景
下，后者的进一步防御收益很低，做了也拦不住调试器/root）。密文承载格式很简单：``cryptography``
的 ``AESGCM`` 已经把认证 tag 附在 ciphertext 尾部返回，不需要手动拆出单独的 ``tag`` 字段。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from miku_on_desk.config.settings import EnvBootstrap

_NONCE_LENGTH = 12  # 96-bit，AES-GCM 推荐的 nonce 长度


def _load_or_create_master_key(key_path: Path) -> bytes:
    if key_path.exists():
        return bytes.fromhex(key_path.read_text(encoding="utf-8").strip())
    key = AESGCM.generate_key(bit_length=256)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(key.hex(), encoding="utf-8")
    os.chmod(key_path, 0o600)
    return key


class SecretVault:
    """基于 SQLite 的加密密钥存储；落盘的 ``value`` 列始终是密文，从不是明文。"""

    def __init__(self, db_path: Path, key_path: Path) -> None:
        self._aesgcm = AESGCM(_load_or_create_master_key(key_path))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS secrets ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL, "
            "description TEXT, created_at INTEGER, updated_at INTEGER)"
        )
        self._conn.commit()

    def _encrypt(self, plaintext: str) -> str:
        nonce = os.urandom(_NONCE_LENGTH)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return json.dumps({"nonce": nonce.hex(), "ciphertext": ciphertext.hex()})

    def _decrypt(self, blob: str) -> str:
        data = json.loads(blob)
        nonce = bytes.fromhex(data["nonce"])
        ciphertext = bytes.fromhex(data["ciphertext"])
        return self._aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")

    def store(self, key: str, value: str, description: str = "") -> None:
        now = int(time.time())
        encrypted = self._encrypt(value)
        self._conn.execute(
            "INSERT INTO secrets (key, value, description, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "description=excluded.description, updated_at=excluded.updated_at",
            (key, encrypted, description, now, now),
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM secrets WHERE key = ?", (key,)).fetchone()
        return self._decrypt(row[0]) if row else None

    def delete(self, key: str) -> None:
        self._conn.execute("DELETE FROM secrets WHERE key = ?", (key,))
        self._conn.commit()

    def list_keys(self) -> list[str]:
        rows = self._conn.execute("SELECT key FROM secrets ORDER BY key").fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        self._conn.close()


def default_vault_paths(bootstrap: EnvBootstrap | None = None) -> tuple[Path, Path]:
    """返回 ``(db_path, key_path)``；key 与加密数据分开存放,便于分别控制访问权限。"""
    bootstrap = bootstrap or EnvBootstrap()
    data_dir = bootstrap.resolve_data_dir()
    return data_dir / "secrets.db", data_dir / "secrets.key"
