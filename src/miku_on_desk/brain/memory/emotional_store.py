"""`emotional` 层：整文件 JSON 存储的用户偏好与信任模型。

只做原子读写，不做合并/更新的语义决策（该改哪个叶子、`last_updated` 该设成什么值）——
那是 `extraction.py` 编排层的职责，跟 `semantic_store.py`/`episodic_store.py` 保持一致的
"哑"存储层定位，方便独立测试。

`trust_model.json` 的 `decay_model` 字段只保留 schema 占位，不做时间衰减的实际计算。

`MemorySystem` 会被 UI 线程和 Brain 线程共享同一个实例，公开方法内部用
`threading.RLock()` 互斥，理由同 `base_store.py` 模块文档。
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, cast

_DEFAULT_PREFERENCES: dict[str, Any] = {
    "version": "1.0",
    "last_updated": "",
    "confidence_threshold": 0.75,
}

_DEFAULT_TRUST_MODEL: dict[str, Any] = {
    "version": "1.0",
    "last_updated": "",
    "fact_trust_scores": {},
    "entity_consistency": {},
    "decay_model": {
        "half_life_days": 30,
        "last_access_boost": 0.1,
        "repeated_confirmation_boost": 0.05,
    },
}


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp-{uuid.uuid4().hex}")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


class EmotionalStore:
    """`emotional` 层存储：`preferences.json` + `trust_model.json`，整文件原子重写。"""

    def __init__(self, root: Path, *, default_confidence_threshold: float = 0.75) -> None:
        self._root = root
        self._preferences_path = root / "preferences.json"
        self._trust_model_path = root / "trust_model.json"
        self._default_preferences: dict[str, Any] = {
            **_DEFAULT_PREFERENCES,
            "confidence_threshold": default_confidence_threshold,
        }
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def load_preferences(self) -> dict[str, Any]:
        with self._lock:
            return _read_json(self._preferences_path, self._default_preferences)

    def save_preferences(self, data: dict[str, Any]) -> None:
        with self._lock:
            _write_json(self._preferences_path, data)

    def load_trust_model(self) -> dict[str, Any]:
        with self._lock:
            return _read_json(self._trust_model_path, _DEFAULT_TRUST_MODEL)

    def save_trust_model(self, data: dict[str, Any]) -> None:
        with self._lock:
            _write_json(self._trust_model_path, data)
