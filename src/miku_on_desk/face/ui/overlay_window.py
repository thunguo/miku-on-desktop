"""无边框透明置顶窗口：承载一个 2D 精灵图小部件，用状态机驱动帧动画。

窗口本身不知道"某个状态该播哪一帧"的细节——``PetStateMachine`` 只关心状态与时间的
纯函数合成，``PetSpriteWidget`` 只知道怎么画一帧,真正把两者粘起来的是这里的
``_on_animation_tick``。事件来源有两路,都最终落到同一个状态机上：Brain 的 9 种事件
（``bridge/events.py``）与外部 CLI 工具的 hook 通知（``face/hooks``）,分别由
``_on_brain_event``/``_on_hook_event`` 消费。``_on_hook_event`` 还额外把事件流喂给
``SessionTracker``，在会话边界（详见 ``face/hooks/session_report.py``）到达时生成一段
战报小结,与 ``PetStateMachine`` 的状态切换完全独立、互不影响。
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QContextMenuEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from miku_on_desk.brain.providers.base import ToolUseBlock
from miku_on_desk.bridge.events import (
    EXPRESS_REACTION_TOOL_NAME,
    AcpChunkReceived,
    BrainCrashed,
    BrainEvent,
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
from miku_on_desk.face.hooks.schema import HookEvent, TransitionKind, resolve_transition
from miku_on_desk.face.hooks.session_report import (
    CompanionGrowth,
    GrowthStore,
    SessionReport,
    SessionTracker,
    format_session_report,
    growth_flavor_text,
    update_growth,
)
from miku_on_desk.face.pet_motion import PetTargetWalker, PetWalker, compute_stand_position
from miku_on_desk.face.pet_state import PetState, PetStateMachine
from miku_on_desk.face.sprite_sheet import SpriteSheetMeta, frame_index
from miku_on_desk.face.stt_worker import SttWorker
from miku_on_desk.face.ui.audio_capture import PcmAudioCapture
from miku_on_desk.face.ui.chat_popup import ChatPopup
from miku_on_desk.face.ui.radial_menu import RadialMenu
from miku_on_desk.face.ui.speech_bubble import SpeechBubble
from miku_on_desk.face.ui.speech_controller import SpeechController
from miku_on_desk.face.ui.sprite_widget import PetSpriteWidget
from miku_on_desk.face.ui.theme import HOVER_COLOR, PINK_ACCENT, PRESSED_COLOR, RADIUS_SM, TEAL_DARK

if TYPE_CHECKING:
    from miku_on_desk.main import PetActions

logger = logging.getLogger(__name__)

_BUBBLE_MARGIN = 10
# 状态/帧推进定时器：30fps 对离散精灵帧切换来说绰绰有余，不需要跟随显示器刷新率。
_ANIMATION_TICK_MS = 33
# 流式增量（ContentDelta/AcpChunkReceived）到达频率可能远超人眼可感知的重排速度，
# 每条都做一次整窗 setGeometry 会造成气泡明显抖动/闪烁——合并到与动画同频的节流窗口内。
_REFLOW_THROTTLE_MS = 33
# Stop/StopFailure 代表一整轮外部会话结束，除了播放一次性反应外还要把常驻基线收回 IDLE，
# 与 Brain 侧 LoopFinished 的处理方式对称——不能只靠 resolve_transition 的查表结果，
# 因为查表只知道"这个事件对应哪个 transient"，不知道"这个事件还标志着一轮交互彻底结束"。
# AfterAgent 是 Gemini CLI 里语义对应的"一轮 agent 交互结束"事件，同样需要收回 baseline。
_HOOK_EVENTS_RESETTING_BASELINE = frozenset({"Stop", "StopFailure", "AfterAgent"})

# express_reaction 工具的反应词表。
_REACTION_STATE_MAP: dict[ReactionKind, PetState] = {
    ReactionKind.HAPPY: PetState.SUCCESS,
    ReactionKind.SAD: PetState.ERROR,
    ReactionKind.SURPRISED: PetState.CLICKED,
    ReactionKind.CURIOUS: PetState.NOTICE,
}

_STOP_BUTTON_SIZE = 22
_STOP_BUTTON_MARGIN = 6
_STOP_BUTTON_STYLE = f"""
QPushButton {{
    background-color: {PINK_ACCENT};
    border: 2px solid {TEAL_DARK};
    border-radius: {RADIUS_SM}px;
    color: white;
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: {HOVER_COLOR};
}}
QPushButton:pressed {{
    background-color: {PRESSED_COLOR};
}}
"""

# acp_delegate/spawn_agents 天然可能跑几分钟到十几分钟——这两个工具名之外的普通工具（读文件、
# 跑命令等）通常几秒内返回，展示耗时标签反而是噪音，所以只对这个白名单显示进度标签。
_LONG_TASK_TOOL_NAMES = frozenset({"acp_delegate", "spawn_agents"})
# computer_input 的 click 分支 requires_confirmation=True 是无条件的（即使 trusted_mode
# 也不豁免），所以 ConfirmationRequested 一定会在点击真正执行前带着 x/y 到达——这是唯一
# 可靠的"提前知道点击目标、点击还没发生"的时机，用来触发"走过去但不挡住"的动画。
_COMPUTER_INPUT_TOOL_NAME = "computer_input"
_PROGRESS_LABEL_MARGIN = 6
_PROGRESS_UPDATE_INTERVAL_MS = 1000
_PROGRESS_LABEL_STYLE = f"""
QLabel {{
    background-color: {TEAL_DARK};
    color: white;
    border-radius: {RADIUS_SM}px;
    padding: 2px 6px;
    font-size: 11px;
}}
"""


class OverlayWindow(QWidget):
    """无边框透明置顶窗口，拖动窗口任意位置即可移动；承载一个 ``PetSpriteWidget``。

    ``event_bus``/``confirmation_gate``/``cancellation_gate``/``hook_bus`` 均为可选：不传时
    窗口只是一个纯展示 spike。都传入时，Brain 事件与外部 CLI hook 事件共享同一个
    ``PetStateMachine``，谁先到就先生效，互不冲突（transient 后来者覆盖、baseline 只在真正
    变化时重置计时）；``cancellation_gate`` 单独驱动任务进行中显示的停止按钮。``growth_store``
    同样可选：不传时仍会展示单次会话的战报小结，只是不追加/持久化跨会话的心情曲线。
    """

    def __init__(
        self,
        pet_dir: Path,
        x: int = 100,
        y: int = 100,
        scale: float = 1.0,
        always_on_top: bool = True,
        walk_enabled: bool = True,
        event_bus: BrainEventBus | None = None,
        confirmation_gate: ConfirmationGate | None = None,
        cancellation_gate: CancellationGate | None = None,
        hook_bus: HookEventBus | None = None,
        actions: PetActions | None = None,
        speech_controller: SpeechController | None = None,
        voice_capture: PcmAudioCapture | None = None,
        stt_worker: SttWorker | None = None,
        growth_store: GrowthStore | None = None,
    ) -> None:
        super().__init__()
        meta = SpriteSheetMeta.load(pet_dir / "pet.json")
        sheet_path = pet_dir / "spritesheet.png"
        self._meta = meta
        self._actions = actions
        self._scale = scale

        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        if always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        if sys.platform == "darwin":
            # macOS 上 Qt::Tool 会映射到 NSPanel 的浮动/工具窗样式，默认在宿主 App
            # 失去焦点时自动隐藏——这个属性是 Qt 官方提供的对应解法，让窗口在失焦后
            # 仍然常驻显示。Windows 的 Qt::Tool 没有这种失焦自动隐藏行为，不需要处理。
            self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._walker: PetWalker | None = PetWalker() if walk_enabled else None
        self._target_walker: PetTargetWalker | None = PetTargetWalker() if walk_enabled else None
        self._pending_click_target: tuple[int, int] | None = None
        self._pending_click_tool_use_id: str | None = None

        self._sprite_widget = PetSpriteWidget(meta, sheet_path, scale=scale, parent=self)

        self._bubble = SpeechBubble(self)
        self._bubble.decision_made.connect(self._on_bubble_decision)

        self._cancellation_gate = cancellation_gate
        self._stop_button = QPushButton("■", self)
        self._stop_button.setFixedSize(_STOP_BUTTON_SIZE, _STOP_BUTTON_SIZE)
        self._stop_button.setStyleSheet(_STOP_BUTTON_STYLE)
        self._stop_button.clicked.connect(self._on_stop_clicked)
        self._stop_button.hide()

        self._progress_label = QLabel(self)
        self._progress_label.setStyleSheet(_PROGRESS_LABEL_STYLE)
        self._progress_label.hide()

        self._reflow_bubble()
        self._position_stop_button()
        self.move(x, y)

        self._state_machine = PetStateMachine()
        self._start_time = time.monotonic()
        self._drag_origin: QPoint | None = None
        self._press_pos: QPoint | None = None
        self._dragged = False

        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._on_animation_tick)
        self._animation_timer.start(_ANIMATION_TICK_MS)

        self._reflow_pending = False

        self._confirmation_gate = confirmation_gate
        self._speech_controller = speech_controller
        self._audio_level = 0.0
        self._talking_segment_started_at: float | None = None
        self._talking_active_elapsed = 0.0
        self._talking_last_tick_t = 0.0
        self._connect_speech_controller_audio_level(speech_controller)
        self._voice_capture = voice_capture
        self._stt_worker = stt_worker
        self._session_tracker = SessionTracker()
        self._growth_store = growth_store
        self._growth = growth_store.load() if growth_store is not None else CompanionGrowth()
        self._pending_confirmation_request_id: str | None = None
        self._tool_use_names: dict[str, str] = {}
        self._acp_active_agent: str | None = None
        self._long_task_tool_use_id: str | None = None
        self._long_task_name: str | None = None
        self._long_task_started_at: float | None = None
        self._progress_timer = QTimer(self)
        self._progress_timer.timeout.connect(self._update_progress_label)
        if event_bus is not None:
            event_bus.brain_event.connect(self._on_brain_event)
        if hook_bus is not None:
            hook_bus.hook_event.connect(self._on_hook_event)

    def _position_bubble(self) -> None:
        width = max(self.width() - 2 * _BUBBLE_MARGIN, 0)
        height = self._bubble.ideal_height(width)
        self._bubble.setGeometry(_BUBBLE_MARGIN, _BUBBLE_MARGIN, width, height)

    def _reflow_bubble(self) -> None:
        """在气泡文字/确认态变化后调用：按新内容重新计算气泡高度，连带调整宿主窗口的
        总高度与位置，使精灵底部在屏幕上的绝对位置保持不变——气泡只在精灵头顶向上
        长大/缩小，不会因为窗口跟着变高而把精灵一起往下推。
        """
        width = max(self._sprite_widget.width() - 2 * _BUBBLE_MARGIN, 0)
        bubble_height = self._bubble.ideal_height(width)
        reserved = bubble_height + _BUBBLE_MARGIN
        new_total_height = reserved + self._sprite_widget.height()
        bottom = self.y() + self.height()
        self.setGeometry(
            self.x(), bottom - new_total_height, self._sprite_widget.width(), new_total_height
        )
        self._sprite_widget.move(0, reserved)
        self._position_bubble()

    def _schedule_reflow(self) -> None:
        """流式增量高频到达时的节流入口：文字已经在调用方立即追加进气泡，这里只把
        `_reflow_bubble` 的整窗 `setGeometry` 合并到每 `_REFLOW_THROTTLE_MS` 至多一次，
        避免连续多条 delta 造成明显的窗口抖动/闪烁。
        """
        if self._reflow_pending:
            return
        self._reflow_pending = True
        QTimer.singleShot(_REFLOW_THROTTLE_MS, self._do_scheduled_reflow)

    def _do_scheduled_reflow(self) -> None:
        self._reflow_pending = False
        self._reflow_bubble()

    def _elapsed(self) -> float:
        return time.monotonic() - self._start_time

    def _connect_speech_controller_audio_level(
        self, speech_controller: SpeechController | None
    ) -> None:
        if speech_controller is not None:
            speech_controller.audio_level_changed.connect(self._on_audio_level_changed)

    def _on_audio_level_changed(self, level: float) -> None:
        self._audio_level = level

    def _advance_talking_elapsed(self, t: float, entered_at: float) -> float:
        """把 TALKING 帧号推进的"经过时间"从墙钟时间换成"响度加权的有效播放时间"：
        响度 0（静音/没有语音功能）时完全不推进，帧定格在当前一帧；响度 1 时按精灵表
        原始 fps 正常循环；中间值按比例放慢，不会比原速更快。
        """
        if entered_at != self._talking_segment_started_at:
            self._talking_segment_started_at = entered_at
            self._talking_active_elapsed = 0.0
            self._talking_last_tick_t = t
            return 0.0
        dt = t - self._talking_last_tick_t
        self._talking_last_tick_t = t
        self._talking_active_elapsed += dt * self._audio_level
        return self._talking_active_elapsed

    def _on_animation_tick(self) -> None:
        t = self._elapsed()
        state = self._state_machine.current_state(t)
        entered_at = self._state_machine.state_entered_at(t)
        info = self._meta.states.get(state, self._meta.states[self._meta.fallback_state])
        if state == PetState.TALKING and self._speech_controller is not None:
            elapsed_in_state = self._advance_talking_elapsed(t, entered_at)
        else:
            elapsed_in_state = t - entered_at
        frame = frame_index(
            elapsed_in_state, fps=info.fps, frame_count=info.frame_count, loop=info.loop
        )
        self._sprite_widget.set_frame(state, frame)

        if self._target_walker is not None and self._pending_click_target is not None:
            screen = self.screen()
            if screen is not None:
                avail = screen.availableGeometry()
                screen_rect = (
                    avail.x(),
                    avail.y(),
                    avail.x() + avail.width(),
                    avail.y() + avail.height(),
                )
                old_x, old_y = self.x(), self.y()
                stand = compute_stand_position(
                    *self._pending_click_target, self.width(), self.height(), screen_rect, old_x
                )
                new_x, new_y = self._target_walker.tick(t, old_x, old_y, stand)
                if (new_x, new_y) != (old_x, old_y):
                    self.move(new_x, new_y)
                    if new_x != old_x:
                        self._sprite_widget.set_facing(new_x > old_x)
        elif self._walker is not None and state == PetState.IDLE:
            screen = self.screen()
            if screen is not None:
                avail = screen.availableGeometry()
                bounds = (avail.x(), avail.x() + avail.width() - self.width())
                new_x = self._walker.tick(t, self.x(), bounds)
                if new_x != self.x():
                    self.move(new_x, self.y())
                self._sprite_widget.set_facing(self._walker.facing_right)

    def _on_brain_event(self, event: BrainEvent) -> None:
        t = self._elapsed()
        if isinstance(event, ContentDelta):
            self._acp_active_agent = None
            self._bubble.append_speech(event.text)
            if self._speech_controller is not None:
                self._speech_controller.feed(event.text)
            self._schedule_reflow()
            self._state_machine.set_baseline_state(PetState.TALKING, t=t)
            self._show_stop_button()
        elif isinstance(event, ThinkingDelta):
            self._state_machine.set_baseline_state(PetState.THINKING, t=t)
            self._show_stop_button()
        elif isinstance(event, AcpChunkReceived):
            if self._acp_active_agent != event.agent:
                self._acp_active_agent = event.agent
                self._bubble.append_speech(f"\n[{event.agent}] ")
            self._bubble.append_speech(event.text)
            self._schedule_reflow()
            self._show_stop_button()
        elif isinstance(event, ToolUseStarted):
            self._tool_use_names[event.tool_use.id] = event.tool_use.name
            if event.tool_use.name != EXPRESS_REACTION_TOOL_NAME:
                self._state_machine.set_baseline_state(PetState.TOOL_RUNNING, t=t)
            if event.tool_use.name in _LONG_TASK_TOOL_NAMES:
                self._start_long_task_progress(event.tool_use.id, event.tool_use.name, t)
            self._show_stop_button()
        elif isinstance(event, ToolResultReceived):
            tool_name = self._tool_use_names.pop(event.result.tool_use_id, None)
            if tool_name != EXPRESS_REACTION_TOOL_NAME:
                state = PetState.ERROR if event.result.is_error else PetState.SUCCESS
                self._state_machine.trigger_transient(state, t=t)
            if event.result.tool_use_id == self._long_task_tool_use_id:
                self._stop_long_task_progress()
            if event.result.tool_use_id == self._pending_click_tool_use_id:
                self._clear_pending_click_target()
        elif isinstance(event, ConfirmationRequested):
            if (
                self._pending_confirmation_request_id is not None
                and self._pending_confirmation_request_id != event.request_id
                and self._confirmation_gate is not None
            ):
                # 罕见边界情况：acp_delegate/spawn_agents 的并发子代理可能各自触发一次
                # 确认请求——气泡一次只能追踪一个 request_id，旧请求若被静默覆盖，其
                # Future 永远等不到 resolve，会让对应的 Brain 任务永久挂起。保守起见
                # 直接拒绝旧请求，避免协程悬挂/网关泄漏，同时记录日志便于排查。
                logger.warning(
                    "收到新的确认请求 %s，但前一个请求 %s 尚未被处理，已自动拒绝前一个请求",
                    event.request_id,
                    self._pending_confirmation_request_id,
                )
                self._confirmation_gate.resolve(self._pending_confirmation_request_id, False)
            self._pending_confirmation_request_id = event.request_id
            self._state_machine.set_baseline_state(PetState.CONFIRMATION_PENDING, t=t)
            self._bubble.show_confirmation(event.reason or f'是否允许 "{event.tool_use.name}"？')
            self._reflow_bubble()
            self._show_stop_button()
            self._maybe_start_click_target_walk(event.tool_use)
        elif isinstance(event, QueuedMessageInjected):
            self._state_machine.trigger_transient(PetState.NOTICE, t=t)
        elif isinstance(event, ReactionTriggered):
            self._state_machine.trigger_transient(_REACTION_STATE_MAP[event.kind], t=t)
        elif isinstance(event, LoopFinished):
            self._acp_active_agent = None
            if self._speech_controller is not None:
                self._speech_controller.flush()
            if event.result.error is not None:
                self._state_machine.trigger_transient(PetState.ERROR, t=t)
            self._state_machine.set_baseline_state(PetState.IDLE, t=t)
            if self._bubble.is_awaiting_confirmation():
                self._bubble.clear()
                self._pending_confirmation_request_id = None
            self._stop_long_task_progress()
            self._stop_button.hide()
            self._stop_button.setEnabled(True)
            self._clear_pending_click_target()
        elif isinstance(event, BrainRestarting):
            self._state_machine.trigger_transient(PetState.NOTICE, t=t)
            self._bubble.show_speech(
                f"呀，内部出了点小问题，正在自动恢复中……"
                f"（第 {event.attempt}/{event.max_attempts} 次尝试）"
            )
            self._reflow_bubble()
        elif isinstance(event, BrainCrashed):
            self._acp_active_agent = None
            if self._speech_controller is not None:
                self._speech_controller.stop()
            self._state_machine.set_baseline_state(PetState.ERROR, t=t)
            self._bubble.show_speech(
                f"呀……我的大脑好像出问题停止了：\n{event.error}\n（需要重启应用才能恢复）"
            )
            self._reflow_bubble()
            self._stop_long_task_progress()
            self._stop_button.hide()
            self._clear_pending_click_target()

    def _show_stop_button(self) -> None:
        self._stop_button.show()
        self._stop_button.raise_()

    def _start_long_task_progress(self, tool_use_id: str, name: str, t: float) -> None:
        self._long_task_tool_use_id = tool_use_id
        self._long_task_name = name
        self._long_task_started_at = t
        self._update_progress_label()
        self._progress_label.show()
        self._progress_label.raise_()
        self._progress_timer.start(_PROGRESS_UPDATE_INTERVAL_MS)

    def _stop_long_task_progress(self) -> None:
        self._progress_timer.stop()
        self._progress_label.hide()
        self._long_task_tool_use_id = None
        self._long_task_name = None
        self._long_task_started_at = None

    def _maybe_start_click_target_walk(self, tool_use: ToolUseBlock) -> None:
        if self._target_walker is None or tool_use.name != _COMPUTER_INPUT_TOOL_NAME:
            return
        if tool_use.input.get("action") != "click":
            return
        x, y = tool_use.input.get("x"), tool_use.input.get("y")
        if not isinstance(x, int | float) or not isinstance(y, int | float):
            return
        self._pending_click_target = (int(x), int(y))
        self._pending_click_tool_use_id = tool_use.id
        self._target_walker.reset()

    def _clear_pending_click_target(self) -> None:
        self._pending_click_target = None
        self._pending_click_tool_use_id = None

    def _update_progress_label(self) -> None:
        if self._long_task_started_at is None or self._long_task_name is None:
            return
        elapsed = self._elapsed() - self._long_task_started_at
        self._progress_label.setText(f"{self._long_task_name} · {elapsed:.0f}s")
        self._position_progress_label()

    def _position_progress_label(self) -> None:
        self._progress_label.adjustSize()
        x = self.width() - self._progress_label.width() - _PROGRESS_LABEL_MARGIN
        y = _STOP_BUTTON_MARGIN + _STOP_BUTTON_SIZE + _PROGRESS_LABEL_MARGIN
        self._progress_label.move(max(x, 0), y)

    def _on_hook_event(self, event: HookEvent) -> None:
        t = self._elapsed()
        report = self._session_tracker.observe(event, t=t)
        if report is not None:
            self._show_session_report(report)

        transition = resolve_transition(event.event)
        if transition is None:
            logger.info("忽略未知 hook 事件：%s", event.event)
            return
        if transition.kind is TransitionKind.BASELINE:
            self._state_machine.set_baseline_state(transition.state, t=t)
            return
        self._state_machine.trigger_transient(transition.state, t=t)
        if event.event in _HOOK_EVENTS_RESETTING_BASELINE:
            self._state_machine.set_baseline_state(PetState.IDLE, t=t)

    def _show_session_report(self, report: SessionReport) -> None:
        """会话边界（``SessionEnd``，或 Codex 场景下的下一次 ``SessionStart``）触发的
        战报小结；心情曲线只在配置了持久化的 ``growth_store`` 时才更新/追加一句话，
        没配置时退化为"只报告这次会话，不追加长期心情"。
        """
        text = format_session_report(report)
        if self._growth_store is not None:
            self._growth = update_growth(self._growth, report)
            flavor = growth_flavor_text(self._growth)
            if flavor is not None:
                text = f"{text}\n{flavor}"
            self._growth_store.save(self._growth)
        self._bubble.show_speech(text)
        self._reflow_bubble()

    def _on_bubble_decision(self, approved: bool) -> None:
        if self._confirmation_gate is None or self._pending_confirmation_request_id is None:
            return
        self._confirmation_gate.resolve(self._pending_confirmation_request_id, approved)
        self._pending_confirmation_request_id = None

    def _position_stop_button(self) -> None:
        x = self.width() - _STOP_BUTTON_SIZE - _STOP_BUTTON_MARGIN
        self._stop_button.move(x, _STOP_BUTTON_MARGIN)


    def set_pet_dir(self, pet_dir: Path) -> None:
        """整体替换精灵图部件切换到另一个角色——``PetSpriteWidget`` 没有热重载方法，
        所有 (状态, 帧) 组合都是构造时预裁剪/预缩放进 ``QPixmap`` 缓存的。

        新建的子 widget 在本窗口早已 ``show()`` 过之后才创建，不会被纳入 Qt 那次一次性
        的显示级联，默认处于隐藏状态——必须显式 ``show()``，否则精灵会"切换成功但看不见"。

        ``deleteLater()`` 是异步删除，真正回收发生在下一次事件循环迭代——如果不显式
        ``hide()`` 旧 widget，两个精灵会在这段窗口期内重叠可见，造成一帧切换闪烁。
        """
        meta = SpriteSheetMeta.load(pet_dir / "pet.json")
        sheet_path = pet_dir / "spritesheet.png"
        old_sprite_widget = self._sprite_widget
        self._sprite_widget = PetSpriteWidget(meta, sheet_path, scale=self._scale, parent=self)
        self._sprite_widget.move(old_sprite_widget.pos())
        self._sprite_widget.show()
        old_sprite_widget.hide()
        old_sprite_widget.deleteLater()
        self._meta = meta
        self._state_machine = PetStateMachine()
        self._start_time = time.monotonic()
        self._reflow_bubble()
        self._position_stop_button()
        self.update()


    def set_speech_controller(self, speech_controller: SpeechController | None) -> None:
        """设置保存后 TTS 配置热重载时同步语音控制器身份（None ↔ 非 None，或整个换新实例）；
        这个属性是构造时赋值一次的普通属性，不像 ``main()`` 里的同名局部变量能靠闭包后绑定
        自动生效，需要显式同步。
        """
        if self._speech_controller is not None:
            self._speech_controller.audio_level_changed.disconnect(self._on_audio_level_changed)
        self._speech_controller = speech_controller
        self._connect_speech_controller_audio_level(speech_controller)

    def set_voice_input(
        self, voice_capture: PcmAudioCapture | None, stt_worker: SttWorker | None
    ) -> None:
        """跟 ``set_speech_controller`` 同理：设置保存后语音输入配置热重载时同步，下一次
        ``_show_chat_popup`` 构造的新 ``ChatPopup`` 会拿到最新值。
        """
        self._voice_capture = voice_capture
        self._stt_worker = stt_worker

    def confirm_via_hotkey(self, approved: bool) -> None:
        """全局热键触发确认（是/否）；复用 _on_bubble_decision 的判定逻辑——
        没有待确认请求时是安全的 no-op。"""
        self._on_bubble_decision(approved)

    def open_chat_via_hotkey(self) -> None:
        """全局热键触发聊天弹窗；复用 _show_chat_popup，没有点击位置可用，
        用窗口自身位置作锚点，跟右键菜单用点击位置/托盘用光标位置类似。"""
        self._show_chat_popup(self.pos())

    def _on_stop_clicked(self) -> None:
        if self._cancellation_gate is not None:
            self._cancellation_gate.request_stop()
        if self._speech_controller is not None:
            self._speech_controller.stop()
        self._stop_button.setEnabled(False)

    def _on_barge_in_requested(self) -> None:
        if self._cancellation_gate is not None:
            self._cancellation_gate.request_stop()
        if self._speech_controller is not None:
            self._speech_controller.stop()

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        self._state_machine.trigger_transient(PetState.CLICKED, t=self._elapsed())
        if self._actions is None:
            return
        actions = self._actions
        global_pos = event.globalPos()
        menu = RadialMenu(self)
        menu.talk_requested.connect(lambda: self._show_chat_popup(global_pos))
        menu.settings_requested.connect(actions.open_settings)
        menu.memory_requested.connect(actions.open_memory)
        menu.recollections_requested.connect(actions.open_recollections)
        menu.characters_requested.connect(actions.open_characters)
        menu.quit_requested.connect(actions.quit)
        menu.popup_at(global_pos)

    def _show_chat_popup(self, global_pos: QPoint) -> None:
        popup = ChatPopup(self, voice_capture=self._voice_capture, stt_worker=self._stt_worker)
        popup.text_submitted.connect(self._route_chat_text)
        popup.barge_in_requested.connect(self._on_barge_in_requested)
        popup.popup_at(global_pos)

    def _route_chat_text(self, text: str) -> None:
        """停止按钮可见即代表当前有 loop 在跑：忙碌时插话入队，空闲时照常直接开始新一轮。

        用 ``isVisibleTo(self)`` 而不是 ``isVisible()``——后者还会级联检查顶层窗口自身
        是否 ``show()`` 过，测试里构造的窗口通常不会真的 show，会导致误判为"一直空闲"。
        """
        assert self._actions is not None
        if self._stop_button.isVisibleTo(self):
            self._actions.queue_message(text)
        else:
            self._actions.talk(text)

    def resizeEvent(self, event: object) -> None:
        self._position_bubble()
        self._position_stop_button()
        self._position_progress_label()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._press_pos = event.globalPosition().toPoint()
        self._drag_origin = self._press_pos - self.pos()
        self._dragged = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin is None or self._press_pos is None:
            return
        current = event.globalPosition().toPoint()
        if not self._dragged:
            moved = (current - self._press_pos).manhattanLength()
            if moved < QApplication.startDragDistance():
                return
            self._dragged = True
            self._state_machine.set_dragging(True, t=self._elapsed())
        self.move(current - self._drag_origin)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin is None:
            return
        if self._dragged:
            self._state_machine.set_dragging(False, t=self._elapsed())
        else:
            self._state_machine.trigger_transient(PetState.CLICKED, t=self._elapsed())
        self._drag_origin = None
        self._press_pos = None
        self._dragged = False
