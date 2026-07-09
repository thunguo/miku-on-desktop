"""OverlayWindow 的接线回归测试：验证 8 种 Brain 事件、hook 事件都能正确路由到共享的
``PetStateMachine``，以及点击/拖拽的鼠标事件判定——不 show() 窗口，不依赖真实生成的
美术资产（用测试内合成的最小 pet.json + spritesheet.png 夹具）。

另外覆盖 macOS 常驻窗口属性（monkeypatch ``sys.platform``）与 ``PetWalker`` 的接线
（monkeypatch ``time.monotonic`` 后直接调用 ``_on_animation_tick``）——真实的失焦不
隐藏效果与走动观感仍需人工在真机上核查，这里只验证接线本身是否正确。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QContextMenuEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from miku_on_desk.brain.loop import LoopResult, LoopStopReason, QueuedMessage
from miku_on_desk.brain.providers.base import ToolResultBlock, ToolUseBlock
from miku_on_desk.bridge.events import (
    AcpChunkReceived,
    BrainCrashed,
    BrainEventBus,
    BrainRestarting,
    CancellationGate,
    ConfirmationGate,
    ConfirmationRequested,
    ContentDelta,
    LoopFinished,
    QueuedMessageInjected,
    ReactionKind,
    ReactionTriggered,
    ThinkingDelta,
    ToolResultReceived,
    ToolUseStarted,
)
from miku_on_desk.face.hooks.bridge import HookEventBus
from miku_on_desk.face.hooks.schema import HookEvent
from miku_on_desk.face.hooks.session_report import GrowthStore
from miku_on_desk.face.pet_state import PetState
from miku_on_desk.face.ui import overlay_window as overlay_window_module
from miku_on_desk.face.ui.chat_popup import ChatPopup
from miku_on_desk.face.ui.overlay_window import OverlayWindow
from miku_on_desk.face.ui.radial_menu import RadialMenu
from miku_on_desk.main import PetActions

_FRAME_SIZE = 4


def _make_pet_dir(
    tmp_path: Path,
    *,
    dir_name: str = "pet",
    pet_name: str = "test_pet",
    frame_size: int = _FRAME_SIZE,
) -> Path:
    pet_dir = tmp_path / dir_name
    pet_dir.mkdir()
    Image.new("RGBA", (frame_size, frame_size), (255, 0, 0, 255)).save(pet_dir / "spritesheet.png")
    meta = {
        "pet_name": pet_name,
        "frame_width": frame_size,
        "frame_height": frame_size,
        "columns": 1,
        "rows": 1,
        "fallback_state": "idle",
        "states": {"idle": {"row": 0, "frame_count": 1, "fps": 1.0, "loop": True}},
    }
    (pet_dir / "pet.json").write_text(json.dumps(meta), encoding="utf-8")
    return pet_dir


def _make_window(tmp_path: Path, **kwargs: object) -> OverlayWindow:
    return OverlayWindow(_make_pet_dir(tmp_path), **kwargs)  # type: ignore[arg-type]


def _mouse_event(kind: QEvent.Type, pos: tuple[int, int]) -> QMouseEvent:
    point = QPointF(*pos)
    return QMouseEvent(
        kind,
        point,
        point,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _context_menu_event(pos: tuple[int, int]) -> QContextMenuEvent:
    point = QPoint(*pos)
    return QContextMenuEvent(QContextMenuEvent.Reason.Mouse, point, point)


def test_content_delta_appends_to_bubble_and_sets_talking_baseline(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)

    bus.emit_event(ContentDelta("你"))
    bus.emit_event(ContentDelta("好"))

    assert window._bubble.current_text() == "你好"
    assert window._state_machine.current_state(window._elapsed()) == PetState.TALKING


def test_thinking_delta_sets_thinking_baseline(qapp: QApplication, tmp_path: Path) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)

    bus.emit_event(ThinkingDelta("嗯…"))

    assert window._state_machine.current_state(window._elapsed()) == PetState.THINKING


def test_acp_chunk_received_appends_to_bubble_with_agent_prefix(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)

    bus.emit_event(AcpChunkReceived(agent="claude-code", text="正在"))
    bus.emit_event(AcpChunkReceived(agent="claude-code", text="重构"))

    assert window._bubble.current_text() == "\n[claude-code] 正在重构"


def test_acp_chunk_received_reprefixes_on_agent_switch(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)

    bus.emit_event(AcpChunkReceived(agent="claude-code", text="任务一"))
    bus.emit_event(AcpChunkReceived(agent="codex", text="任务二"))

    assert window._bubble.current_text() == "\n[claude-code] 任务一\n[codex] 任务二"


def test_content_delta_after_acp_chunk_resets_agent_prefix_tracking(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)

    bus.emit_event(AcpChunkReceived(agent="claude-code", text="任务一"))
    bus.emit_event(ContentDelta("我回来了"))
    bus.emit_event(AcpChunkReceived(agent="claude-code", text="任务二"))

    assert window._bubble.current_text() == "\n[claude-code] 任务一我回来了\n[claude-code] 任务二"


def test_tool_use_started_sets_tool_running_baseline(qapp: QApplication, tmp_path: Path) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)
    tool_use = ToolUseBlock(id="c1", name="computer_input", input={})

    bus.emit_event(ToolUseStarted(tool_use))

    assert window._state_machine.current_state(window._elapsed()) == PetState.TOOL_RUNNING


def test_tool_result_received_success_triggers_success_transient(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)
    result = ToolResultBlock(tool_use_id="c1", content="ok", is_error=False)

    bus.emit_event(ToolResultReceived(result))

    assert window._state_machine.current_state(window._elapsed()) == PetState.SUCCESS


def test_tool_result_received_error_triggers_error_transient(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)
    result = ToolResultBlock(tool_use_id="c1", content="boom", is_error=True)

    bus.emit_event(ToolResultReceived(result))

    assert window._state_machine.current_state(window._elapsed()) == PetState.ERROR


def test_express_reaction_tool_use_does_not_set_tool_running_baseline(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)
    tool_use = ToolUseBlock(id="c1", name="express_reaction", input={"kind": "happy"})

    bus.emit_event(ToolUseStarted(tool_use))

    assert window._state_machine.current_state(window._elapsed()) != PetState.TOOL_RUNNING


def test_express_reaction_tool_result_does_not_trigger_generic_success_transient(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)
    tool_use = ToolUseBlock(id="c1", name="express_reaction", input={"kind": "happy"})
    result = ToolResultBlock(tool_use_id="c1", content="ok", is_error=False)

    bus.emit_event(ToolUseStarted(tool_use))
    bus.emit_event(ToolResultReceived(result))

    assert window._state_machine.current_state(window._elapsed()) != PetState.SUCCESS


@pytest.mark.parametrize("tool_name", ["acp_delegate", "spawn_agents"])
def test_long_task_tool_use_started_shows_progress_label(
    qapp: QApplication, tmp_path: Path, tool_name: str
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)
    tool_use = ToolUseBlock(id="c1", name=tool_name, input={})

    bus.emit_event(ToolUseStarted(tool_use))

    assert window._progress_label.isVisibleTo(window) is True
    assert tool_name in window._progress_label.text()


def test_short_task_tool_use_started_does_not_show_progress_label(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)
    tool_use = ToolUseBlock(id="c1", name="computer_input", input={})

    bus.emit_event(ToolUseStarted(tool_use))

    assert window._progress_label.isVisibleTo(window) is False


def test_long_task_tool_result_received_hides_progress_label(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)
    tool_use = ToolUseBlock(id="c1", name="acp_delegate", input={})
    result = ToolResultBlock(tool_use_id="c1", content="ok", is_error=False)

    bus.emit_event(ToolUseStarted(tool_use))
    bus.emit_event(ToolResultReceived(result))

    assert window._progress_label.isVisibleTo(window) is False


def test_loop_finished_hides_progress_label_even_without_matching_tool_result(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)
    tool_use = ToolUseBlock(id="c1", name="spawn_agents", input={})

    bus.emit_event(ToolUseStarted(tool_use))
    bus.emit_event(
        LoopFinished(
            LoopResult(stop_reason=LoopStopReason.USER_CANCELLED, messages=[], rounds=0)
        )
    )

    assert window._progress_label.isVisibleTo(window) is False


@pytest.mark.parametrize(
    ("kind", "expected_state"),
    [
        (ReactionKind.HAPPY, PetState.SUCCESS),
        (ReactionKind.SAD, PetState.ERROR),
        (ReactionKind.SURPRISED, PetState.CLICKED),
        (ReactionKind.CURIOUS, PetState.NOTICE),
    ],
)
def test_reaction_triggered_maps_kind_to_expected_transient(
    qapp: QApplication, tmp_path: Path, kind: ReactionKind, expected_state: PetState
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)

    bus.emit_event(ReactionTriggered(kind=kind))

    assert window._state_machine.current_state(window._elapsed()) == expected_state


async def test_confirmation_requested_shows_confirmation_and_decision_resolves_gate(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    gate = ConfirmationGate(bus)
    window = _make_window(tmp_path, event_bus=bus, confirmation_gate=gate)
    tool_use = ToolUseBlock(id="c1", name="computer_input", input={})

    async def click_yes_once_prompted() -> None:
        for _ in range(50):
            if window._bubble.is_awaiting_confirmation():
                window._bubble._yes_button.click()
                return
            await asyncio.sleep(0)
        pytest.fail("确认气泡没有在预期时间内出现")

    approved, _ = await asyncio.gather(
        gate.request(tool_use, "要点击吗？"), click_yes_once_prompted()
    )

    assert approved is True
    assert window._bubble.current_text() == "要点击吗？"
    assert window._pending_confirmation_request_id is None
    assert window._state_machine.current_state(window._elapsed()) == PetState.CONFIRMATION_PENDING


def test_confirmation_requested_without_reason_falls_back_to_tool_name(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)
    tool_use = ToolUseBlock(id="c1", name="computer_input", input={})

    bus.emit_event(ConfirmationRequested("r1", tool_use, None))

    assert window._bubble.current_text() == '是否允许 "computer_input"？'


async def test_confirmation_requested_while_previous_unresolved_auto_denies_stale_request(
    qapp: QApplication, tmp_path: Path
) -> None:
    """acp_delegate/spawn_agents 的并发子代理可能各自触发一次确认请求——气泡一次只能
    追踪一个 request_id，旧请求若被静默覆盖，其 Future 永远等不到 resolve，会让对应的
    Brain 任务永久挂起。新请求到达时应自动拒绝旧请求，避免协程悬挂/网关泄漏。
    """
    bus = BrainEventBus()
    gate = ConfirmationGate(bus)
    window = _make_window(tmp_path, event_bus=bus, confirmation_gate=gate)
    tool_use_a = ToolUseBlock(id="a", name="acp_delegate", input={})
    tool_use_b = ToolUseBlock(id="b", name="acp_delegate", input={})

    first_task = asyncio.ensure_future(gate.request(tool_use_a, "第一个确认"))
    for _ in range(50):
        if window._pending_confirmation_request_id is not None:
            break
        await asyncio.sleep(0)
    else:
        pytest.fail("第一个确认请求没有在预期时间内到达")
    first_request_id = window._pending_confirmation_request_id

    bus.emit_event(ConfirmationRequested("second", tool_use_b, "第二个确认"))
    first_approved = await first_task

    assert first_approved is False
    assert first_request_id != "second"
    assert window._pending_confirmation_request_id == "second"


async def test_yes_shortcut_activated_approves_pending_confirmation(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    gate = ConfirmationGate(bus)
    window = _make_window(tmp_path, event_bus=bus, confirmation_gate=gate)
    tool_use = ToolUseBlock(id="c1", name="computer_input", input={})

    async def press_yes_once_prompted() -> None:
        for _ in range(50):
            if window._bubble.is_awaiting_confirmation():
                window.confirm_via_hotkey(True)
                return
            await asyncio.sleep(0)
        pytest.fail("确认气泡没有在预期时间内出现")

    approved, _ = await asyncio.gather(
        gate.request(tool_use, "要点击吗？"), press_yes_once_prompted()
    )

    assert approved is True
    assert window._pending_confirmation_request_id is None


async def test_no_shortcut_activated_rejects_pending_confirmation(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    gate = ConfirmationGate(bus)
    window = _make_window(tmp_path, event_bus=bus, confirmation_gate=gate)
    tool_use = ToolUseBlock(id="c1", name="computer_input", input={})

    async def press_no_once_prompted() -> None:
        for _ in range(50):
            if window._bubble.is_awaiting_confirmation():
                window.confirm_via_hotkey(False)
                return
            await asyncio.sleep(0)
        pytest.fail("确认气泡没有在预期时间内出现")

    approved, _ = await asyncio.gather(
        gate.request(tool_use, "要点击吗？"), press_no_once_prompted()
    )

    assert approved is False
    assert window._pending_confirmation_request_id is None


def test_shortcut_activation_without_pending_confirmation_is_noop(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    gate = ConfirmationGate(bus)
    window = _make_window(tmp_path, event_bus=bus, confirmation_gate=gate)

    window.confirm_via_hotkey(True)
    window.confirm_via_hotkey(False)

    assert window._pending_confirmation_request_id is None


def test_open_chat_via_hotkey_shows_chat_popup(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created_popups: list[ChatPopup] = []
    monkeypatch.setattr(ChatPopup, "popup_at", lambda self, global_pos: created_popups.append(self))
    window = _make_window(tmp_path)

    window.open_chat_via_hotkey()

    assert len(created_popups) == 1


def test_queued_message_injected_triggers_notice_transient(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)

    bus.emit_event(QueuedMessageInjected(QueuedMessage(queued_id="q1", text="插一句")))

    assert window._state_machine.current_state(window._elapsed()) == PetState.NOTICE


def test_loop_finished_without_error_resets_baseline_to_idle(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)
    bus.emit_event(ContentDelta("你"))

    bus.emit_event(LoopFinished(LoopResult(stop_reason=LoopStopReason.DONE, messages=[], rounds=1)))

    assert window._state_machine.current_state(window._elapsed()) == PetState.IDLE


def test_loop_finished_resets_acp_agent_prefix_tracking(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)
    bus.emit_event(AcpChunkReceived(agent="claude-code", text="任务一"))

    bus.emit_event(LoopFinished(LoopResult(stop_reason=LoopStopReason.DONE, messages=[], rounds=1)))
    bus.emit_event(AcpChunkReceived(agent="claude-code", text="任务二"))

    assert window._bubble.current_text() == "\n[claude-code] 任务一\n[claude-code] 任务二"


def test_loop_finished_with_error_triggers_error_transient(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)

    bus.emit_event(
        LoopFinished(
            LoopResult(
                stop_reason=LoopStopReason.PROVIDER_ERROR,
                messages=[],
                rounds=1,
                error="炸了",
            )
        )
    )

    assert window._state_machine.current_state(window._elapsed()) == PetState.ERROR


@pytest.mark.parametrize(
    "event",
    [
        ContentDelta("你"),
        ThinkingDelta("嗯…"),
        AcpChunkReceived(agent="claude-code", text="进行中"),
        ToolUseStarted(ToolUseBlock(id="c1", name="computer_input", input={})),
        ConfirmationRequested("r1", ToolUseBlock(id="c1", name="computer_input", input={}), None),
    ],
)
def test_task_in_progress_events_show_stop_button(
    qapp: QApplication, tmp_path: Path, event: object
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)

    bus.emit_event(event)  # type: ignore[arg-type]

    assert window._stop_button.isVisibleTo(window) is True


def test_loop_finished_hides_stop_button_and_reenables_it(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)
    bus.emit_event(ContentDelta("你"))
    window._stop_button.setEnabled(False)

    bus.emit_event(LoopFinished(LoopResult(stop_reason=LoopStopReason.DONE, messages=[], rounds=1)))

    assert window._stop_button.isVisibleTo(window) is False
    assert window._stop_button.isEnabled() is True


def test_loop_finished_after_confirmation_requested_clears_stale_confirmation_bubble(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)
    tool_use = ToolUseBlock(id="c1", name="computer_input", input={})
    bus.emit_event(ConfirmationRequested("r1", tool_use, "要点击吗？"))
    assert window._bubble.is_awaiting_confirmation() is True

    bus.emit_event(
        LoopFinished(
            LoopResult(stop_reason=LoopStopReason.USER_CANCELLED, messages=[], rounds=0)
        )
    )

    assert window._bubble.is_awaiting_confirmation() is False
    assert window._pending_confirmation_request_id is None


def test_brain_restarting_shows_transient_notice(qapp: QApplication, tmp_path: Path) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)

    bus.emit_event(BrainRestarting(attempt=1, max_attempts=3, delay_s=1.5, error="炸了"))

    assert window._state_machine.current_state(window._elapsed()) == PetState.NOTICE
    assert "1/3" in window._bubble.current_text()


def test_brain_crashed_sets_error_baseline_and_shows_message(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)

    bus.emit_event(BrainCrashed(error="炸了"))

    assert window._state_machine.current_state(window._elapsed()) == PetState.ERROR
    assert "炸了" in window._bubble.current_text()
    assert window._bubble.is_awaiting_confirmation() is False


def test_brain_crashed_hides_stop_button(qapp: QApplication, tmp_path: Path) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)
    bus.emit_event(ContentDelta("你"))
    assert window._stop_button.isVisibleTo(window) is True

    bus.emit_event(BrainCrashed(error="炸了"))

    assert window._stop_button.isVisibleTo(window) is False


def test_stop_button_click_invokes_cancellation_gate_request_stop(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[None] = []
    gate = CancellationGate()
    monkeypatch.setattr(gate, "request_stop", lambda: requested.append(None))
    window = _make_window(tmp_path, cancellation_gate=gate)

    window._stop_button.click()

    assert requested == [None]
    assert window._stop_button.isEnabled() is False


def test_stop_button_click_without_cancellation_gate_is_noop(
    qapp: QApplication, tmp_path: Path
) -> None:
    window = _make_window(tmp_path)

    window._stop_button.click()

    assert window._stop_button.isEnabled() is False


def test_window_without_event_bus_ignores_events_safely(qapp: QApplication, tmp_path: Path) -> None:
    window = _make_window(tmp_path)

    assert window._bubble.current_text() == ""


def test_resize_repositions_bubble_to_span_new_width(qapp: QApplication, tmp_path: Path) -> None:
    window = _make_window(tmp_path)

    window.resize(800, 900)
    window.resizeEvent(None)

    assert window._bubble.width() == 800 - 20


def test_content_delta_growing_bubble_keeps_sprite_bottom_anchored(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: fn())
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)
    sprite_bottom_before = window.y() + window._sprite_widget.y() + window._sprite_widget.height()

    bus.emit_event(ContentDelta("这是一段很长的文字，需要占用不止一行才能完整显示 " * 10))

    sprite_bottom_after = window.y() + window._sprite_widget.y() + window._sprite_widget.height()
    assert sprite_bottom_after == sprite_bottom_before
    assert window._sprite_widget.y() > 0


def test_content_delta_batches_rapid_reflow_calls_into_a_single_scheduled_timer(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """流式增量到达频率可能远超肉眼可感知的重排速度——多条连续 delta 应该只调度一次
    延迟重排，而不是每条都立刻做一次整窗 setGeometry（那样会造成明显抖动/闪烁）。
    """
    scheduled: list[int] = []
    monkeypatch.setattr(QTimer, "singleShot", lambda ms, _fn: scheduled.append(ms))
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)

    bus.emit_event(ContentDelta("你"))
    bus.emit_event(ContentDelta("好"))
    bus.emit_event(ContentDelta("呀"))

    assert len(scheduled) == 1
    assert window._reflow_pending is True
    assert window._bubble.current_text() == "你好呀"


def test_scheduled_reflow_resets_pending_flag_so_next_delta_reschedules(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduled: list[int] = []
    monkeypatch.setattr(QTimer, "singleShot", lambda ms, fn: (scheduled.append(ms), fn())[1])
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)

    bus.emit_event(ContentDelta("你"))
    assert window._reflow_pending is False
    assert len(scheduled) == 1

    bus.emit_event(ContentDelta("好"))
    assert len(scheduled) == 2


def test_acp_chunk_received_also_throttles_reflow(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduled: list[int] = []
    monkeypatch.setattr(QTimer, "singleShot", lambda ms, _fn: scheduled.append(ms))
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus)

    bus.emit_event(AcpChunkReceived(agent="claude-code", text="正在"))
    bus.emit_event(AcpChunkReceived(agent="claude-code", text="重构"))

    assert len(scheduled) == 1
    assert window._reflow_pending is True


def test_hook_event_baseline_transition_sets_baseline_state(
    qapp: QApplication, tmp_path: Path
) -> None:
    hook_bus = HookEventBus()
    window = _make_window(tmp_path, hook_bus=hook_bus)

    hook_bus.emit_event(HookEvent(event="UserPromptSubmit"))

    assert window._state_machine.current_state(window._elapsed()) == PetState.THINKING


def test_hook_event_transient_transition_triggers_transient(
    qapp: QApplication, tmp_path: Path
) -> None:
    hook_bus = HookEventBus()
    window = _make_window(tmp_path, hook_bus=hook_bus)

    hook_bus.emit_event(HookEvent(event="PostToolUseFailure", tool_name="Bash", error="boom"))

    assert window._state_machine.current_state(window._elapsed()) == PetState.ERROR


def test_hook_event_stop_resets_baseline_to_idle_after_transient(
    qapp: QApplication, tmp_path: Path
) -> None:
    hook_bus = HookEventBus()
    window = _make_window(tmp_path, hook_bus=hook_bus)
    hook_bus.emit_event(HookEvent(event="UserPromptSubmit"))

    hook_bus.emit_event(HookEvent(event="Stop"))

    assert window._state_machine._baseline_state == PetState.IDLE
    assert window._state_machine.current_state(window._elapsed()) == PetState.SUCCESS


def test_hook_event_after_agent_resets_baseline_to_idle_after_transient(
    qapp: QApplication, tmp_path: Path
) -> None:
    hook_bus = HookEventBus()
    window = _make_window(tmp_path, hook_bus=hook_bus)
    hook_bus.emit_event(HookEvent(event="BeforeAgent"))

    hook_bus.emit_event(HookEvent(event="AfterAgent"))

    assert window._state_machine._baseline_state == PetState.IDLE
    assert window._state_machine.current_state(window._elapsed()) == PetState.SUCCESS


def test_hook_event_unknown_event_is_ignored(qapp: QApplication, tmp_path: Path) -> None:
    hook_bus = HookEventBus()
    window = _make_window(tmp_path, hook_bus=hook_bus)

    hook_bus.emit_event(HookEvent(event="SomeFutureEvent"))

    assert window._state_machine.current_state(window._elapsed()) == PetState.IDLE


def test_hook_event_session_end_shows_session_report_in_bubble(
    qapp: QApplication, tmp_path: Path
) -> None:
    hook_bus = HookEventBus()
    window = _make_window(tmp_path, hook_bus=hook_bus)

    hook_bus.emit_event(HookEvent(event="SessionStart", source="claude_code"))
    hook_bus.emit_event(HookEvent(event="PostToolUse"))
    hook_bus.emit_event(HookEvent(event="SessionEnd"))

    assert "次工具" in window._bubble.current_text()


def test_hook_event_turn_level_stop_does_not_trigger_session_report(
    qapp: QApplication, tmp_path: Path
) -> None:
    hook_bus = HookEventBus()
    window = _make_window(tmp_path, hook_bus=hook_bus)

    hook_bus.emit_event(HookEvent(event="SessionStart", source="claude_code"))
    hook_bus.emit_event(HookEvent(event="UserPromptSubmit"))
    hook_bus.emit_event(HookEvent(event="Stop"))

    assert window._bubble.current_text() == ""


def test_hook_event_codex_style_session_start_finalizes_previous_session(
    qapp: QApplication, tmp_path: Path
) -> None:
    """Codex CLI 没有安装 SessionEnd,靠下一次 SessionStart 补记上一个会话的战报。"""
    hook_bus = HookEventBus()
    window = _make_window(tmp_path, hook_bus=hook_bus)

    hook_bus.emit_event(HookEvent(event="SessionStart", source="codex"))
    hook_bus.emit_event(HookEvent(event="PostToolUse"))
    assert window._bubble.current_text() == ""

    hook_bus.emit_event(HookEvent(event="SessionStart", source="codex"))

    assert "次工具" in window._bubble.current_text()


def test_hook_event_session_report_without_growth_store_has_no_growth_flavor(
    qapp: QApplication, tmp_path: Path
) -> None:
    hook_bus = HookEventBus()
    window = _make_window(tmp_path, hook_bus=hook_bus)

    hook_bus.emit_event(HookEvent(event="SessionStart"))
    hook_bus.emit_event(HookEvent(event="SessionEnd"))

    assert "第 1 次" not in window._bubble.current_text()


def test_hook_event_session_report_with_growth_store_adds_milestone_flavor_and_persists(
    qapp: QApplication, tmp_path: Path
) -> None:
    hook_bus = HookEventBus()
    growth_path = tmp_path / "companion_growth.json"
    store = GrowthStore(growth_path)
    window = _make_window(tmp_path, hook_bus=hook_bus, growth_store=store)

    hook_bus.emit_event(HookEvent(event="SessionStart"))
    hook_bus.emit_event(HookEvent(event="SessionEnd"))

    assert "第 1 次" in window._bubble.current_text()
    assert store.load().sessions_completed == 1


def test_click_without_drag_triggers_clicked_transient(qapp: QApplication, tmp_path: Path) -> None:
    window = _make_window(tmp_path)

    window.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, (10, 10)))
    window.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, (10, 10)))

    assert window._state_machine.current_state(window._elapsed()) == PetState.CLICKED
    assert window._dragged is False


def test_drag_beyond_threshold_sets_dragging_and_release_clears_it(
    qapp: QApplication, tmp_path: Path
) -> None:
    window = _make_window(tmp_path)
    origin = window.pos()

    window.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, (10, 10)))
    window.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, (50, 50)))

    assert window._dragged is True
    assert window._state_machine.current_state(window._elapsed()) == PetState.DRAGGED
    assert window.pos() != origin

    window.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, (50, 50)))

    assert window._state_machine.current_state(window._elapsed()) != PetState.DRAGGED


def test_mac_platform_sets_always_show_tool_window_attribute(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(overlay_window_module.sys, "platform", "darwin")

    window = _make_window(tmp_path)

    assert window.testAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow) is True


def test_non_mac_platform_does_not_set_always_show_tool_window_attribute(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(overlay_window_module.sys, "platform", "win32")

    window = _make_window(tmp_path)

    assert window.testAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow) is False


def test_walk_moves_window_horizontally_while_baseline_is_idle(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"t": 0.0}
    monkeypatch.setattr(overlay_window_module.time, "monotonic", lambda: clock["t"])
    window = _make_window(tmp_path, walk_enabled=True)
    origin_x = window.x()

    window._on_animation_tick()
    clock["t"] = 1.0
    window._on_animation_tick()

    assert window.x() != origin_x


def test_walk_does_not_move_window_while_baseline_is_not_idle(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"t": 0.0}
    monkeypatch.setattr(overlay_window_module.time, "monotonic", lambda: clock["t"])
    window = _make_window(tmp_path, walk_enabled=True)
    window._state_machine.set_baseline_state(PetState.TALKING, t=window._elapsed())
    origin_x = window.x()

    window._on_animation_tick()
    clock["t"] = 1.0
    window._on_animation_tick()

    assert window.x() == origin_x


def test_walk_enabled_false_never_moves_window(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"t": 0.0}
    monkeypatch.setattr(overlay_window_module.time, "monotonic", lambda: clock["t"])
    window = _make_window(tmp_path, walk_enabled=False)
    origin_x = window.x()

    window._on_animation_tick()
    clock["t"] = 1.0
    window._on_animation_tick()

    assert window.x() == origin_x
    assert window._walker is None


def test_confirmation_requested_click_action_sets_pending_click_target(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus, walk_enabled=True)
    tool_use = ToolUseBlock(
        id="c1", name="computer_input", input={"action": "click", "x": 500, "y": 500}
    )

    bus.emit_event(ConfirmationRequested("r1", tool_use, None))

    assert window._pending_click_target == (500, 500)
    assert window._pending_click_tool_use_id == "c1"


@pytest.mark.parametrize("action", ["type_text", "key_press", "open_app"])
def test_confirmation_requested_non_click_action_does_not_set_pending_click_target(
    qapp: QApplication, tmp_path: Path, action: str
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus, walk_enabled=True)
    tool_use = ToolUseBlock(id="c1", name="computer_input", input={"action": action})

    bus.emit_event(ConfirmationRequested("r1", tool_use, None))

    assert window._pending_click_target is None


def test_confirmation_requested_other_tool_name_does_not_set_pending_click_target(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus, walk_enabled=True)
    tool_use = ToolUseBlock(
        id="c1", name="read_file", input={"action": "click", "x": 500, "y": 500}
    )

    bus.emit_event(ConfirmationRequested("r1", tool_use, None))

    assert window._pending_click_target is None


def test_confirmation_requested_click_action_with_non_numeric_coordinates_is_ignored(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus, walk_enabled=True)
    tool_use = ToolUseBlock(
        id="c1", name="computer_input", input={"action": "click", "x": "not-a-number", "y": 500}
    )

    bus.emit_event(ConfirmationRequested("r1", tool_use, None))

    assert window._pending_click_target is None


def test_confirmation_requested_click_walk_disabled_does_not_set_pending_click_target(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus, walk_enabled=False)
    tool_use = ToolUseBlock(
        id="c1", name="computer_input", input={"action": "click", "x": 500, "y": 500}
    )

    bus.emit_event(ConfirmationRequested("r1", tool_use, None))

    assert window._target_walker is None
    assert window._pending_click_target is None


def test_pending_click_target_moves_window_during_confirmation_pending_baseline(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"t": 0.0}
    monkeypatch.setattr(overlay_window_module.time, "monotonic", lambda: clock["t"])
    window = _make_window(tmp_path, walk_enabled=True)
    window._state_machine.set_baseline_state(PetState.CONFIRMATION_PENDING, t=window._elapsed())
    window._pending_click_target = (window.x() + 500, window.y())
    origin_x = window.x()

    window._on_animation_tick()
    clock["t"] = 1.0
    window._on_animation_tick()

    assert window.x() != origin_x
    assert window._walker is not None
    assert window._walker._last_t is None


def test_tool_result_received_matching_id_clears_pending_click_target(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus, walk_enabled=True)
    tool_use = ToolUseBlock(
        id="c1", name="computer_input", input={"action": "click", "x": 500, "y": 500}
    )
    bus.emit_event(ConfirmationRequested("r1", tool_use, None))
    assert window._pending_click_target is not None

    bus.emit_event(
        ToolResultReceived(ToolResultBlock(tool_use_id="c1", content="ok", is_error=False))
    )

    assert window._pending_click_target is None
    assert window._pending_click_tool_use_id is None


def test_tool_result_received_non_matching_id_does_not_clear_pending_click_target(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus, walk_enabled=True)
    tool_use = ToolUseBlock(
        id="c1", name="computer_input", input={"action": "click", "x": 500, "y": 500}
    )
    bus.emit_event(ConfirmationRequested("r1", tool_use, None))

    bus.emit_event(
        ToolResultReceived(ToolResultBlock(tool_use_id="other", content="ok", is_error=False))
    )

    assert window._pending_click_target == (500, 500)


def test_loop_finished_clears_pending_click_target_unconditionally(
    qapp: QApplication, tmp_path: Path
) -> None:
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus, walk_enabled=True)
    tool_use = ToolUseBlock(
        id="c1", name="computer_input", input={"action": "click", "x": 500, "y": 500}
    )
    bus.emit_event(ConfirmationRequested("r1", tool_use, None))
    assert window._pending_click_target is not None

    bus.emit_event(
        LoopFinished(LoopResult(stop_reason=LoopStopReason.DONE, messages=[], rounds=1))
    )

    assert window._pending_click_target is None
    assert window._pending_click_tool_use_id is None


def test_idle_wander_resumes_after_pending_click_target_cleared(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"t": 0.0}
    monkeypatch.setattr(overlay_window_module.time, "monotonic", lambda: clock["t"])
    bus = BrainEventBus()
    window = _make_window(tmp_path, event_bus=bus, walk_enabled=True)
    tool_use = ToolUseBlock(
        id="c1", name="computer_input", input={"action": "click", "x": 500, "y": 500}
    )
    bus.emit_event(ConfirmationRequested("r1", tool_use, None))
    window._on_animation_tick()

    bus.emit_event(
        LoopFinished(LoopResult(stop_reason=LoopStopReason.DONE, messages=[], rounds=1))
    )
    origin_x = window.x()

    clock["t"] = 1.0
    window._on_animation_tick()
    clock["t"] = 2.0
    window._on_animation_tick()

    assert window.x() != origin_x
    assert window._pending_click_target is None


def test_context_menu_without_actions_does_nothing(qapp: QApplication, tmp_path: Path) -> None:
    window = _make_window(tmp_path)

    window.contextMenuEvent(_context_menu_event((10, 10)))

    assert window._state_machine.current_state(window._elapsed()) == PetState.CLICKED


def test_context_menu_wires_radial_menu_signals_to_actions(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created_menus: list[RadialMenu] = []
    created_popups: list[ChatPopup] = []
    monkeypatch.setattr(RadialMenu, "popup_at", lambda self, global_pos: created_menus.append(self))
    monkeypatch.setattr(ChatPopup, "popup_at", lambda self, global_pos: created_popups.append(self))

    talked: list[str] = []
    queued: list[str] = []
    settings_calls: list[None] = []
    memory_calls: list[None] = []
    characters_calls: list[None] = []
    quit_calls: list[None] = []
    actions = PetActions(
        talk=talked.append,
        queue_message=queued.append,
        open_settings=lambda: settings_calls.append(None),
        open_memory=lambda: memory_calls.append(None),
        open_characters=lambda: characters_calls.append(None),
        quit=lambda: quit_calls.append(None),
    )
    window = _make_window(tmp_path, actions=actions)

    window.contextMenuEvent(_context_menu_event((10, 10)))

    assert len(created_menus) == 1
    menu = created_menus[0]
    menu.settings_requested.emit()
    menu.memory_requested.emit()
    menu.characters_requested.emit()
    menu.quit_requested.emit()
    menu.talk_requested.emit()

    assert settings_calls == [None]
    assert memory_calls == [None]
    assert characters_calls == [None]
    assert quit_calls == [None]
    assert len(created_popups) == 1

    created_popups[0].text_submitted.emit("你好")

    assert talked == ["你好"]
    assert queued == []


def test_context_menu_chat_popup_routes_to_queue_message_when_busy(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """停止按钮可见（有 loop 在跑）时，聊天气泡提交的文本应该插话入队而不是直接开始新一轮。"""
    created_popups: list[ChatPopup] = []
    monkeypatch.setattr(ChatPopup, "popup_at", lambda self, global_pos: created_popups.append(self))

    talked: list[str] = []
    queued: list[str] = []
    actions = PetActions(
        talk=talked.append,
        queue_message=queued.append,
        open_settings=lambda: None,  # type: ignore[arg-type]
        open_memory=lambda: None,  # type: ignore[arg-type]
        open_characters=lambda: None,  # type: ignore[arg-type]
        quit=lambda: None,
    )
    window = _make_window(tmp_path, actions=actions)
    window.show()
    window._show_stop_button()

    window._show_chat_popup(QPoint(10, 10))
    assert len(created_popups) == 1
    created_popups[0].text_submitted.emit("插话")

    assert queued == ["插话"]
    assert talked == []


def test_set_pet_dir_replaces_sprite_widget_and_resets_state_machine(
    qapp: QApplication, tmp_path: Path
) -> None:
    other_frame_size = _FRAME_SIZE * 2
    other_pet_dir = _make_pet_dir(
        tmp_path, dir_name="other_pet", pet_name="other_pet", frame_size=other_frame_size
    )
    window = _make_window(tmp_path)
    original_sprite_widget = window._sprite_widget

    window.set_pet_dir(other_pet_dir)

    assert window._meta.pet_name == "other_pet"
    assert window._sprite_widget is not original_sprite_widget
    assert window._sprite_widget.width() == other_frame_size
    assert window.width() == other_frame_size



def test_set_speech_controller_replaces_internal_reference(
    qapp: QApplication, tmp_path: Path
) -> None:
    """设置保存后的 TTS 热重载靠这个 setter 同步身份——``_speech_controller`` 是构造时
    赋值一次的普通属性，不会像 ``main()`` 里的同名局部变量一样自动跟着后绑定生效。
    """
    window = _make_window(tmp_path)
    new_controller = Mock()

    window.set_speech_controller(new_controller)

    assert window._speech_controller is new_controller


def test_set_pet_dir_new_sprite_widget_is_visible_after_window_shown(
    qapp: QApplication, tmp_path: Path
) -> None:
    """新精灵是在窗口已经 show() 过之后创建的子 widget，不会被 Qt 的显示级联覆盖，
    必须显式 show()——否则会出现"切换成功但看不见，要重启才显示"的问题。
    """
    other_pet_dir = _make_pet_dir(tmp_path, dir_name="other_pet", pet_name="other_pet")
    window = _make_window(tmp_path)
    window.show()

    window.set_pet_dir(other_pet_dir)

    assert window._sprite_widget.isVisibleTo(window) is True


def test_set_pet_dir_hides_old_sprite_widget_immediately_to_avoid_overlap_flash(
    qapp: QApplication, tmp_path: Path
) -> None:
    """``deleteLater()`` 是异步删除，真正回收发生在下一次事件循环迭代——如果不显式
    ``hide()`` 旧精灵，热切换期间新旧两个精灵会短暂重叠可见，造成一帧闪烁。
    """
    other_pet_dir = _make_pet_dir(tmp_path, dir_name="other_pet", pet_name="other_pet")
    window = _make_window(tmp_path)
    window.show()
    original_sprite_widget = window._sprite_widget

    window.set_pet_dir(other_pet_dir)

    assert original_sprite_widget.isVisibleTo(window) is False


def test_stop_button_and_progress_label_have_non_empty_stylesheets(
    qapp: QApplication, tmp_path: Path
) -> None:
    window = _make_window(tmp_path)

    assert window._stop_button.styleSheet().strip() != ""
    assert window._progress_label.styleSheet().strip() != ""

