"""演示 ``eval_llm_judge`` marker 与"无 API key 时显式 skip、不是裸失败或挂起"这一约定；
本身不发起任何真实网络调用，只验证 skip 机制能跑到底。
"""

from __future__ import annotations

import pytest

from tests.eval.judge import requires_llm_judge


@pytest.mark.eval_llm_judge
@requires_llm_judge
def test_llm_judge_gate_skips_cleanly_without_api_key() -> None:
    assert True
