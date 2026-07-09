"""Victory declaration bias：需要用户确认的高风险操作执行成功后，模型容易直接假设"做完了就是
达成目标"，跳过验证就宣布任务完成。Phase 4 计划在这类工具成功后追加 ``[verify-before-done]``
提醒，要求先做一次只读检查再宣布完成——本用例在 Phase 4 落地前应为红灯，Phase 4 交付时必须
转绿。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from miku_on_desk.brain.loop import LoopCallbacks, LoopStopReason, run_ai_loop
from miku_on_desk.brain.providers.base import Message, StreamResult, ToolUseBlock
from miku_on_desk.config.settings import ProviderName
from tests.support.loop_fixtures import (
    SESSION,
    TIER,
    FakeProvider,
    build_router,
    make_confirmation_registry,
    tool_results_by_id,
    tool_use,
)


async def _approve(_tool_use: ToolUseBlock, _reason: str | None) -> bool:
    return True


@pytest.mark.eval_capability
async def test_capability_verify_before_done_reminder_after_confirmed_tool_success(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1", name="dangerous_tool")]),
            StreamResult(success=True, content="done"),
        ]
    )
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_confirmation_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=_approve),
    )
    assert result.stop_reason == LoopStopReason.DONE
    results = tool_results_by_id(result.messages)
    assert results["1"].is_error is False
    assert "[verify-before-done]" in results["1"].content, (
        "需要用户确认的高风险工具执行成功后，预期尾部出现 [verify-before-done] 提醒，但没有"
        "——Phase 4 落地后本用例应转绿"
    )
