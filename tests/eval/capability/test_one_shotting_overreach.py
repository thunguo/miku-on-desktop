"""One-shotting overreach：连续跑很多轮工具调用而不跟用户汇报进度、确认方向，是三类已知失败
模式之一。Phase 4 计划给 ``run_ai_loop`` 加一个默认开启的 ``[progress-checkin]`` 周期性提醒
（默认每 8 轮一次）——本用例在 Phase 4 落地前应为红灯，Phase 4 交付时必须转绿。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from miku_on_desk.brain.loop import LoopCallbacks, LoopConfig, LoopStopReason, run_ai_loop
from miku_on_desk.brain.providers.base import Message, StreamResult
from miku_on_desk.config.settings import ProviderName
from tests.support.loop_fixtures import (
    SESSION,
    TIER,
    FakeProvider,
    build_router,
    make_registry,
    never_confirm,
    tool_results_by_id,
    tool_use,
)


@pytest.mark.eval_capability
async def test_capability_progress_checkin_reminder_fires_during_long_tool_chains(
    tmp_path: Path,
) -> None:
    rounds = 10
    scripted = [StreamResult(success=True, tool_uses=[tool_use(str(i))]) for i in range(rounds)]
    provider = FakeProvider([*scripted, StreamResult(success=True, content="done")])
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="做一件需要很多步骤的事")],
        callbacks=LoopCallbacks(confirm=never_confirm),
        config=LoopConfig(max_tool_rounds=rounds + 5),
    )
    assert result.stop_reason == LoopStopReason.DONE
    results = tool_results_by_id(result.messages)
    checkin_markers = [r.content for r in results.values() if "[progress-checkin]" in r.content]
    assert checkin_markers, (
        f"连续跑了 {rounds} 轮工具调用，预期在第 8 轮左右出现 [progress-checkin] 提醒，但没有"
        "——Phase 4 落地后本用例应转绿"
    )
