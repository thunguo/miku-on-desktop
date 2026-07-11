"""角色关系状态：每个角色目录名 -> 熟悉度整数，跨进程持久化。

写法与 ``session_report.py`` 的 ``GrowthStore`` 一致——原子写入（tmp 文件 + ``os.replace``），
读写失败只记日志、回退到空字典/放弃保存。这是装饰性的关系反馈，不是核心功能，不应该因为这个
文件的问题连带影响角色切换本身。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_RELATIONSHIP_FILE_VERSION = 1


class RelationshipStore:
    """``character_relationships.json`` 的原子读写外壳。"""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> dict[str, int]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            familiarity = data.get("familiarity", {})
            return {str(name): int(value) for name, value in familiarity.items()}
        except Exception:
            logger.exception("读取角色关系文件失败，回退到空字典：%s", self._path)
            return {}

    def save(self, familiarity: dict[str, int]) -> None:
        payload = {
            "version": _RELATIONSHIP_FILE_VERSION,
            "familiarity": familiarity,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(f"{self._path.suffix}.tmp-{uuid.uuid4().hex}")
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(tmp_path, self._path)
        except Exception:
            logger.exception("保存角色关系文件失败，跳过：%s", self._path)

    def get(self, pet_name: str) -> int:
        return self.load().get(pet_name, 0)

    def bump(self, pet_name: str) -> int:
        familiarity = self.load()
        new_value = familiarity.get(pet_name, 0) + 1
        familiarity[pet_name] = new_value
        self.save(familiarity)
        return new_value
