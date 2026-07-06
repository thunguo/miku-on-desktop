"""``scripts/generate_pet_sprites.py`` CLI 入口的回归测试。

`scripts/` 不是 `miku_on_desk` 包的一部分（故意不注册为 console-script），用
`importlib` 按文件路径加载模块，避免通过 `sys.path` 插入污染其他测试的导入环境。

确定性后处理纯函数（帧切割/透明清理/调色板量化/拼装/QA）的测试见
`tests/test_character_generation.py`，直接测试共享实现所在的
`miku_on_desk.character_generation` 模块。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "generate_pet_sprites.py"
    spec = importlib.util.spec_from_file_location("generate_pet_sprites", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gps = _load_script_module()


def test_main_returns_nonzero_without_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = gps.main(["--output-dir", str(tmp_path)])

    assert exit_code == 1


sys.modules.pop("generate_pet_sprites", None)
