"""Hook sidecar 的 bearer token：每次进程启动都重新生成并轮换,而不是 load-or-create——
就是要让上一次进程遗留的 token 立刻失效。0600 权限从 ``os.open`` 创建那一刻就带上,
避免 write-then-chmod 之间出现短暂的宽松权限窗口，沿用
``brain/secrets/vault.py`` 已确立的本地密钥文件权限约定。
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

_TOKEN_NBYTES = 32


def generate_token() -> str:
    return secrets.token_hex(_TOKEN_NBYTES)


def write_token(token_path: Path, token: str) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode("utf-8"))
    finally:
        os.close(fd)


def rotate_token(token_path: Path) -> str:
    """生成一个新 token、写入磁盘并返回它；每次进程启动应调用且只调用一次。"""
    token = generate_token()
    write_token(token_path, token)
    return token
