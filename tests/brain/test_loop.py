"""run_ai_loop 的回归测试：用假 Provider 驱动多轮 流式↔工具执行 往复，验证核心循环条件、
预算软性提醒、以及两个可选注入点（compact_context/consume_queued_message）在各自子系统
缺席（``None``）和存在时的行为。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from miku_on_desk.brain import loop as loop_module
from miku_on_desk.brain.loop import (
    LoopCallbacks,
    LoopConfig,
    LoopStopReason,
    QueuedMessage,
    run_ai_loop,
)
from miku_on_desk.brain.model_router import ModelRouter
from miku_on_desk.brain.providers.base import (
    Message,
    StreamResult,
    StreamUsage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from miku_on_desk.brain.tracing import TRACE_LOGGER_NAME
from miku_on_desk.config.settings import ModelRouterConfig, ProviderName
from tests.support.loop_fixtures import (
    SESSION,
    TIER,
    FakeProvider,
    build_router,
    build_router_with_fallback,
    make_confirmation_registry,
    make_registry,
    never_confirm,
    tool_results_by_id,
    tool_use,
)


async def test_run_ai_loop_returns_done_when_first_response_has_no_tool_calls(
    tmp_path: Path,
) -> None:
    provider = FakeProvider([StreamResult(success=True, content="全部完成")])
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="你好")],
        callbacks=LoopCallbacks(confirm=never_confirm),
    )
    assert result.stop_reason == LoopStopReason.DONE
    assert result.rounds == 0
    assert len(provider.calls) == 1
    assert result.messages[-1].role == "assistant"


async def test_run_ai_loop_executes_tool_then_completes(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1")]),
            StreamResult(success=True, content="完成"),
        ]
    )
    tool_use_events: list[ToolUseBlock] = []
    tool_result_events: list[ToolResultBlock] = []
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="帮我回声")],
        callbacks=LoopCallbacks(
            confirm=never_confirm,
            on_tool_use=tool_use_events.append,
            on_tool_result=tool_result_events.append,
        ),
    )
    assert result.stop_reason == LoopStopReason.DONE
    assert result.rounds == 1
    assert len(provider.calls) == 2
    assert [event.id for event in tool_use_events] == ["1"]
    assert tool_result_events[0].content == "ok"
    assert tool_result_events[0].is_error is False


async def test_run_ai_loop_emits_structured_log_fields(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("DEBUG", logger="miku_on_desk.brain.loop")
    provider = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1")]),
            StreamResult(success=True, content="完成"),
        ]
    )
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="帮我回声")],
        callbacks=LoopCallbacks(confirm=never_confirm),
    )
    assert result.stop_reason == LoopStopReason.DONE
    messages = [record.getMessage() for record in caplog.records]
    assert any("ai_loop loop_start session_id=session-1" in m for m in messages)
    assert any("ai_loop round_start session_id=session-1 round=0" in m for m in messages)
    assert any(
        "ai_loop tool_call_start session_id=session-1 tool=echo_tool" in m for m in messages
    )
    assert any("ai_loop tool_call_end session_id=session-1 tool=echo_tool" in m for m in messages)
    assert any(
        "ai_loop loop_end session_id=session-1 stop_reason=done rounds=1" in m for m in messages
    )


async def test_run_ai_loop_emits_trace_events_for_a_full_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """结构化 trace 与上面的人类可读文本日志完全并行——同一次跑动，分别用两种手法各自
    断言各自的输出，互不干扰、互不派生。"""
    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
    original_propagate = trace_logger.propagate
    trace_logger.propagate = True
    caplog.set_level(logging.DEBUG, logger=TRACE_LOGGER_NAME)
    try:
        provider = FakeProvider(
            [
                StreamResult(
                    success=True,
                    tool_uses=[tool_use("1")],
                    usage=StreamUsage(input_tokens=123, output_tokens=45),
                ),
                StreamResult(success=True, content="完成"),
            ]
        )
        result = await run_ai_loop(
            session_id=SESSION,
            tier=TIER,
            router=build_router(),
            providers={ProviderName.ANTHROPIC: provider},
            registry=make_registry(tmp_path),
            system="sys",
            messages=[Message(role="user", content="帮我回声")],
            callbacks=LoopCallbacks(confirm=never_confirm),
        )
    finally:
        trace_logger.propagate = original_propagate

    assert result.stop_reason == LoopStopReason.DONE
    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == TRACE_LOGGER_NAME
    ]
    events_by_type = [event["event"] for event in events]
    for expected in (
        "loop_start",
        "tool_call_start",
        "tool_call_end",
        "provider_call",
        "loop_end",
    ):
        assert expected in events_by_type

    provider_call_events = [event for event in events if event["event"] == "provider_call"]
    assert provider_call_events[0]["input_tokens"] == 123


async def test_run_ai_loop_stops_with_budget_exhausted_when_rounds_run_out(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1")]),
            StreamResult(success=True, tool_uses=[tool_use("2")]),
        ]
    )
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
        config=LoopConfig(max_tool_rounds=1),
    )
    assert result.stop_reason == LoopStopReason.BUDGET_EXHAUSTED
    assert result.rounds == 1
    assert len(provider.calls) == 2


async def test_run_ai_loop_returns_provider_error_on_initial_call(tmp_path: Path) -> None:
    provider = FakeProvider(
        [StreamResult(success=False, error="provider_error", raw_error="boom")]
    )
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
    )
    assert result.stop_reason == LoopStopReason.PROVIDER_ERROR
    assert result.rounds == 0
    assert result.error == "provider_error"
    assert result.raw_error == "boom"


async def test_run_ai_loop_returns_provider_error_on_subsequent_call(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1")]),
            StreamResult(success=False, error="request_idle_timeout", raw_error="idle"),
        ]
    )
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
    )
    assert result.stop_reason == LoopStopReason.PROVIDER_ERROR
    assert result.rounds == 1
    assert result.error == "模型响应中断，可能是网络问题"
    assert result.raw_error == "idle"


@pytest.mark.parametrize(
    ("error_token", "expected_text"),
    [
        ("request_idle_timeout", "模型响应中断，可能是网络问题"),
        ("request_hard_timeout", "单次请求耗时过长，已强制中断"),
    ],
)
async def test_run_ai_loop_translates_timeout_kind_into_actionable_text(
    tmp_path: Path, error_token: str, expected_text: str
) -> None:
    provider = FakeProvider([StreamResult(success=False, error=error_token, raw_error="raw")])
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
    )
    assert result.error == expected_text


async def test_run_ai_loop_leaves_non_timeout_error_tokens_untranslated(tmp_path: Path) -> None:
    provider = FakeProvider([StreamResult(success=False, error="rate_limited", raw_error="raw")])
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
    )
    assert result.error == "rate_limited"


async def test_run_ai_loop_falls_back_to_another_provider_when_primary_fails(
    tmp_path: Path,
) -> None:
    primary = FakeProvider([StreamResult(success=False, error="client_error", raw_error="denied")])
    fallback = FakeProvider([StreamResult(success=True, content="来自备用")])
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router_with_fallback(),
        providers={ProviderName.ANTHROPIC: primary, ProviderName.OPENAI: fallback},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
    )
    assert result.stop_reason == LoopStopReason.DONE
    assert result.messages[-1].role == "assistant"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


async def test_run_ai_loop_logs_fallback_trigger_with_session_id(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING", logger="miku_on_desk.brain.loop")
    primary = FakeProvider([StreamResult(success=False, error="client_error", raw_error="denied")])
    fallback = FakeProvider([StreamResult(success=True, content="来自备用")])
    await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router_with_fallback(),
        providers={ProviderName.ANTHROPIC: primary, ProviderName.OPENAI: fallback},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
    )
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "ai_loop fallback_triggered session_id=session-1 from_provider=anthropic"
        " error=client_error to_provider=openai" in m
        for m in messages
    )


async def test_run_ai_loop_preserves_original_error_when_fallback_also_fails(
    tmp_path: Path,
) -> None:
    primary = FakeProvider(
        [StreamResult(success=False, error="client_error", raw_error="primary-boom")]
    )
    fallback = FakeProvider(
        [StreamResult(success=False, error="client_error", raw_error="fallback-boom")]
    )
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router_with_fallback(),
        providers={ProviderName.ANTHROPIC: primary, ProviderName.OPENAI: fallback},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
    )
    assert result.stop_reason == LoopStopReason.PROVIDER_ERROR
    assert result.raw_error == "primary-boom"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


async def test_run_ai_loop_does_not_fall_back_when_disabled(tmp_path: Path) -> None:
    primary = FakeProvider(
        [StreamResult(success=False, error="client_error", raw_error="primary-boom")]
    )
    fallback = FakeProvider([StreamResult(success=True, content="不该被用到")])
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router_with_fallback(enabled=False),
        providers={ProviderName.ANTHROPIC: primary, ProviderName.OPENAI: fallback},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
    )
    assert result.stop_reason == LoopStopReason.PROVIDER_ERROR
    assert result.raw_error == "primary-boom"
    assert fallback.calls == []


async def test_run_ai_loop_keeps_using_fallback_provider_for_subsequent_rounds(
    tmp_path: Path,
) -> None:
    primary = FakeProvider(
        [StreamResult(success=False, error="client_error", raw_error="primary-boom")]
    )
    fallback = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1")]),
            StreamResult(success=True, content="完成"),
        ]
    )
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router_with_fallback(),
        providers={ProviderName.ANTHROPIC: primary, ProviderName.OPENAI: fallback},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
    )
    assert result.stop_reason == LoopStopReason.DONE
    assert result.rounds == 1
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 2


async def test_run_ai_loop_returns_no_model_available_when_router_cannot_resolve(
    tmp_path: Path,
) -> None:
    provider = FakeProvider([])
    original_messages = [Message(role="user", content="hi")]
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=ModelRouter(ModelRouterConfig()),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=original_messages,
        callbacks=LoopCallbacks(confirm=never_confirm),
    )
    assert result.stop_reason == LoopStopReason.NO_MODEL_AVAILABLE
    assert result.rounds == 0
    assert result.messages == original_messages
    assert provider.calls == []


async def test_run_ai_loop_feeds_deny_result_back_and_continues(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1", name="unknown_tool")]),
            StreamResult(success=True, content="done"),
        ]
    )
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
    )
    assert result.stop_reason == LoopStopReason.DONE
    tool_results = tool_results_by_id(result.messages)
    assert tool_results["1"].is_error is True
    assert "未知工具" in tool_results["1"].content


async def test_run_ai_loop_executes_tool_after_confirm_approves(tmp_path: Path) -> None:
    async def _approve(_tool_use: ToolUseBlock, reason: str | None) -> bool:
        assert reason == "此操作需要用户确认。"
        return True

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
    tool_results = tool_results_by_id(result.messages)
    assert tool_results["1"].is_error is False
    assert tool_results["1"].content.startswith("ok")
    assert "[verify-before-done]" in tool_results["1"].content


async def test_run_ai_loop_records_error_result_when_confirm_rejects(tmp_path: Path) -> None:
    async def _reject(_tool_use: ToolUseBlock, _reason: str | None) -> bool:
        return False

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
        callbacks=LoopCallbacks(confirm=_reject),
    )
    tool_results = tool_results_by_id(result.messages)
    assert tool_results["1"].is_error is True
    assert "拒绝" in tool_results["1"].content


async def test_budget_warning_attached_to_previous_tool_result_when_threshold_crossed(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1")]),
            StreamResult(success=True, tool_uses=[tool_use("2")]),
            StreamResult(success=True, tool_uses=[tool_use("3")]),
            StreamResult(success=True, content="done"),
        ]
    )
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
        config=LoopConfig(
            max_tool_rounds=4, budget_caution_remaining=2, budget_critical_remaining=1
        ),
    )
    assert result.stop_reason == LoopStopReason.DONE

    def _contents(call_index: int) -> list[str]:
        return [
            block.content
            for message in provider.calls[call_index]["messages"]
            if isinstance(message.content, list)
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]

    assert not any("[turn-budget]" in content for content in _contents(2))
    assert any("[turn-budget]" in content for content in _contents(3))


async def test_budget_warning_escalates_and_does_not_repeat_for_same_tier(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1")]),
            StreamResult(success=True, tool_uses=[tool_use("2")]),
            StreamResult(success=True, tool_uses=[tool_use("3")]),
            StreamResult(success=True, tool_uses=[tool_use("4")]),
            StreamResult(success=True, tool_uses=[tool_use("5")]),
            StreamResult(success=True, content="done"),
        ]
    )
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
        config=LoopConfig(
            max_tool_rounds=5, budget_caution_remaining=3, budget_critical_remaining=1
        ),
    )
    assert result.stop_reason == LoopStopReason.DONE
    assert result.rounds == 5

    tool_results = tool_results_by_id(result.messages)
    assert tool_results["1"].content.count("[turn-budget]") == 0
    assert tool_results["2"].content.count("[turn-budget]") == 1
    assert "如实告知" not in tool_results["2"].content
    assert tool_results["3"].content.count("[turn-budget]") == 0
    assert tool_results["4"].content.count("[turn-budget]") == 1
    assert "如实告知" in tool_results["4"].content
    assert tool_results["5"].content.count("[turn-budget]") == 0


async def test_budget_warning_silently_skipped_when_no_tool_result_exists_yet(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1")]),
            StreamResult(success=True, tool_uses=[tool_use("2")]),
        ]
    )
    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
        config=LoopConfig(max_tool_rounds=1, budget_critical_remaining=5),
    )
    assert result.stop_reason == LoopStopReason.BUDGET_EXHAUSTED


async def test_run_ai_loop_stops_with_time_exhausted_and_preserves_partial_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(
        [StreamResult(success=True, content="部分内容", tool_uses=[tool_use("1")])]
    )
    monotonic_values = iter([0.0, 100.0])
    monkeypatch.setattr(loop_module, "monotonic", lambda: next(monotonic_values))

    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
        config=LoopConfig(deadline_s=10.0),
    )

    assert result.stop_reason == LoopStopReason.TIME_EXHAUSTED
    assert result.rounds == 0
    assert len(provider.calls) == 1
    assert any(
        isinstance(block, TextBlock) and block.text == "部分内容"
        for message in result.messages
        if isinstance(message.content, list)
        for block in message.content
    )


async def test_time_warning_attached_to_previous_tool_result_when_threshold_crossed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1")]),
            StreamResult(success=True, tool_uses=[tool_use("2")]),
            StreamResult(success=True, tool_uses=[tool_use("3")]),
            StreamResult(success=True, content="done"),
        ]
    )
    monotonic_values = iter([0.0, 10.0, 40.0, 60.0])
    monkeypatch.setattr(loop_module, "monotonic", lambda: next(monotonic_values))

    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
        config=LoopConfig(
            deadline_s=100.0, time_caution_remaining_s=50.0, time_critical_remaining_s=10.0
        ),
    )
    assert result.stop_reason == LoopStopReason.DONE

    def _contents(call_index: int) -> list[str]:
        return [
            block.content
            for message in provider.calls[call_index]["messages"]
            if isinstance(message.content, list)
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]

    assert not any("[time-budget]" in content for content in _contents(2))
    assert any("[time-budget]" in content for content in _contents(3))


async def test_time_warning_escalates_and_does_not_repeat_for_same_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1")]),
            StreamResult(success=True, tool_uses=[tool_use("2")]),
            StreamResult(success=True, tool_uses=[tool_use("3")]),
            StreamResult(success=True, tool_uses=[tool_use("4")]),
            StreamResult(success=True, tool_uses=[tool_use("5")]),
            StreamResult(success=True, content="done"),
        ]
    )
    monotonic_values = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    monkeypatch.setattr(loop_module, "monotonic", lambda: next(monotonic_values))

    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
        config=LoopConfig(
            deadline_s=6.0, time_caution_remaining_s=3.0, time_critical_remaining_s=1.0
        ),
    )
    assert result.stop_reason == LoopStopReason.DONE
    assert result.rounds == 5

    tool_results = tool_results_by_id(result.messages)
    assert tool_results["1"].content.count("[time-budget]") == 0
    assert tool_results["2"].content.count("[time-budget]") == 1
    assert "如实告知" not in tool_results["2"].content
    assert tool_results["3"].content.count("[time-budget]") == 0
    assert tool_results["4"].content.count("[time-budget]") == 1
    assert "如实告知" in tool_results["4"].content
    assert tool_results["5"].content.count("[time-budget]") == 0


async def test_compact_context_not_invoked_on_first_iteration_but_invoked_afterward(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1")]),
            StreamResult(success=True, tool_uses=[tool_use("2")]),
            StreamResult(success=True, content="done"),
        ]
    )
    compact_calls: list[list[Message]] = []

    async def _compact(messages: list[Message]) -> list[Message] | None:
        compact_calls.append(list(messages))
        return None

    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm, compact_context=_compact),
    )
    assert result.stop_reason == LoopStopReason.DONE
    assert len(compact_calls) == 1


async def test_compact_context_replaces_working_messages_when_it_returns_a_value(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1")]),
            StreamResult(success=True, tool_uses=[tool_use("2")]),
            StreamResult(success=True, content="done"),
        ]
    )

    async def _compact(_messages: list[Message]) -> list[Message] | None:
        return [Message(role="user", content="压缩后的摘要")]

    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm, compact_context=_compact),
    )
    assert result.stop_reason == LoopStopReason.DONE
    round_b_messages = provider.calls[2]["messages"]
    assert round_b_messages[0].content == "压缩后的摘要"


async def test_consume_queued_message_injects_user_message_and_notifies(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1")]),
            StreamResult(success=True, content="done"),
        ]
    )
    queued = QueuedMessage(queued_id="q1", text="插播消息")
    pending: list[QueuedMessage | None] = [queued, None]
    injected: list[QueuedMessage] = []

    def _consume() -> QueuedMessage | None:
        return pending.pop(0)

    result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(
            confirm=never_confirm,
            consume_queued_message=_consume,
            on_queued_message_injected=injected.append,
        ),
    )
    assert result.stop_reason == LoopStopReason.DONE
    assert injected == [queued]
    call1_messages = provider.calls[1]["messages"]
    assert any(m.role == "user" and m.content == "插播消息" for m in call1_messages)


async def test_on_content_and_on_thinking_are_forwarded_to_provider_stream(
    tmp_path: Path,
) -> None:
    provider = FakeProvider([StreamResult(success=True, content="ok")])
    content_chunks: list[str] = []
    thinking_chunks: list[str] = []
    await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(
            confirm=never_confirm,
            on_content=content_chunks.append,
            on_thinking=thinking_chunks.append,
        ),
    )
    assert content_chunks == ["streamed-content"]
    assert thinking_chunks == ["streamed-thinking"]


async def test_run_ai_loop_passes_timeout_config_to_provider(tmp_path: Path) -> None:
    provider = FakeProvider([StreamResult(success=True, content="ok")])
    await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: provider},
        registry=make_registry(tmp_path),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
        config=LoopConfig(idle_timeout_s=5.0, hard_timeout_s=30.0),
    )
    assert provider.calls[0]["idle_timeout_s"] == 5.0
    assert provider.calls[0]["hard_timeout_s"] == 30.0


async def test_run_ai_loop_appends_verify_before_done_reminder_after_confirmed_tool(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    async def _approve(_tool_use: ToolUseBlock, _reason: str | None) -> bool:
        return True

    confirmed_provider = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1", name="dangerous_tool")]),
            StreamResult(success=True, content="done"),
        ]
    )
    confirmed_result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: confirmed_provider},
        registry=make_confirmation_registry(tmp_path_factory.mktemp("confirmed")),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=_approve),
    )
    confirmed_results = tool_results_by_id(confirmed_result.messages)
    assert "[verify-before-done]" in confirmed_results["1"].content

    plain_provider = FakeProvider(
        [
            StreamResult(success=True, tool_uses=[tool_use("1")]),
            StreamResult(success=True, content="done"),
        ]
    )
    plain_result = await run_ai_loop(
        session_id=SESSION,
        tier=TIER,
        router=build_router(),
        providers={ProviderName.ANTHROPIC: plain_provider},
        registry=make_registry(tmp_path_factory.mktemp("plain")),
        system="sys",
        messages=[Message(role="user", content="hi")],
        callbacks=LoopCallbacks(confirm=never_confirm),
    )
    plain_results = tool_results_by_id(plain_result.messages)
    assert "[verify-before-done]" not in plain_results["1"].content


async def test_run_ai_loop_progress_checkin_fires_every_n_rounds_and_not_more(
    tmp_path: Path,
) -> None:
    scripted = [
        StreamResult(success=True, tool_uses=[tool_use(str(i))]) for i in range(1, 18)
    ]
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
        config=LoopConfig(max_tool_rounds=20, checkin_every_n_rounds=8),
    )
    assert result.stop_reason == LoopStopReason.DONE
    tool_results = tool_results_by_id(result.messages)
    checkin_ids = {
        tool_use_id
        for tool_use_id, block in tool_results.items()
        if "[progress-checkin]" in block.content
    }
    assert checkin_ids == {"8", "16"}


async def test_run_ai_loop_progress_checkin_disabled_when_none(tmp_path: Path) -> None:
    scripted = [StreamResult(success=True, tool_uses=[tool_use(str(i))]) for i in range(1, 11)]
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
        config=LoopConfig(max_tool_rounds=15, checkin_every_n_rounds=None),
    )
    assert result.stop_reason == LoopStopReason.DONE
    tool_results = tool_results_by_id(result.messages)
    assert not any("[progress-checkin]" in block.content for block in tool_results.values())
