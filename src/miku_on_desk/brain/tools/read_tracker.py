"""先读后改追踪：write 类工具要求同一会话内先有对应路径的一次 read 记录。

纯内存 dict，不做任何持久化——一个从未读过某文件的会话不能盲改它，但这个约束不需要跨进程
重启存活，会话的读取历史本来就不该比进程寿命更长。
"""

from __future__ import annotations

from pathlib import Path


class ReadTracker:
    def __init__(self) -> None:
        self._read_by_session: dict[str, set[Path]] = {}

    def mark_read(self, session_id: str, path: Path) -> None:
        self._read_by_session.setdefault(session_id, set()).add(path.resolve())

    def has_been_read(self, session_id: str, path: Path) -> bool:
        return path.resolve() in self._read_by_session.get(session_id, set())

    def clear_session(self, session_id: str) -> None:
        self._read_by_session.pop(session_id, None)
