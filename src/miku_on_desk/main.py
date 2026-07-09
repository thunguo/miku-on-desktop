"""程序入口：初始化日志与配置，装配 Brain 后台线程与 Qt 主循环。

Brain 的 AI 循环是纯 asyncio，跑在一个独立的后台线程里；PySide6 事件循环占用主线程。
Brain→UI 方向直接用 ``BrainEventBus.emit_event``（Qt 信号跨线程会自动切换成
``QueuedConnection``，天然线程安全，不需要额外的队列）；UI→Brain 方向（聊天输入/关闭
信号）用一个普通 ``queue.Queue``，Brain 侧用 ``asyncio.to_thread(chat_input.get)`` 阻塞
等待，避免为了偶发的用户输入去写一个轮询循环。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import sys
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCursor
from PySide6.QtWidgets import (
    QApplication,
    QMenu,
    QStyle,
    QSystemTrayIcon,
    QWidget,
)
from qfluentwidgets import MessageBox

from miku_on_desk.brain.acp.manager import AcpManager, register_acp_delegate_tool
from miku_on_desk.brain.agents.manager import AgentManager, AgentProfile, default_agent_manager
from miku_on_desk.brain.agents.spawn import register_spawn_agents_tool
from miku_on_desk.brain.loop import (
    LoopCallbacks,
    LoopConfig,
    LoopResult,
    LoopStopReason,
    run_ai_loop,
)
from miku_on_desk.brain.mcp.host import MCPHost
from miku_on_desk.brain.memory import extraction
from miku_on_desk.brain.memory.models import Entity, Fact, MemoryUnit, MemoryUnitRole
from miku_on_desk.brain.memory.system import MemorySystem, default_memory_system
from miku_on_desk.brain.model_router import ModelRouter
from miku_on_desk.brain.proactive import ProactiveTrigger, run_proactive_scheduler
from miku_on_desk.brain.prompt.frozen_system import FrozenSystemSections, build_frozen_system
from miku_on_desk.brain.prompt.reminder import build_system_reminder, host_shell_descriptor
from miku_on_desk.brain.provider_factory import build_providers
from miku_on_desk.brain.providers.base import Message, Provider, TextBlock
from miku_on_desk.brain.secrets.vault import SecretVault, default_vault_paths
from miku_on_desk.brain.skills.manager import default_skill_manager, register_skill_tool
from miku_on_desk.brain.tools.builtin.computer_input import register_computer_input_tool
from miku_on_desk.brain.tools.builtin.express_reaction import register_express_reaction_tool
from miku_on_desk.brain.tools.builtin.memory_tools import register_memory_tools
from miku_on_desk.brain.tools.builtin.screen_analyze import register_screen_analyze_tool
from miku_on_desk.brain.tools.path_sandbox import default_path_sandbox
from miku_on_desk.brain.tools.policy import default_policy_engine
from miku_on_desk.brain.tools.read_tracker import ReadTracker
from miku_on_desk.brain.tools.registry import ToolRegistry
from miku_on_desk.bridge.events import (
    AcpChunkReceived,
    BrainCrashed,
    BrainEventBus,
    CancellationGate,
    ConfirmationGate,
    LoopFinished,
    QueuedMessageQueue,
    build_loop_callbacks,
)
from miku_on_desk.config import (
    AgentProfileConfig,
    AppSettings,
    EnvBootstrap,
    HookServerConfig,
    ModelTier,
    PersonaConfig,
    ProviderName,
    default_settings_path,
    load_settings_with_vault,
    save_settings_with_vault,
)
from miku_on_desk.config.logging_config import setup_logging
from miku_on_desk.face.character_voice import resolve_tts_config_for_pet
from miku_on_desk.face.hooks.bridge import HookEventBus
from miku_on_desk.face.hooks.installer import default_claude_settings_path, install
from miku_on_desk.face.hooks.server import PET_EVENT_PATH, HookServer
from miku_on_desk.face.ui.character_clone_dialog import CharacterCloneDialog
from miku_on_desk.face.ui.character_creation_dialog import CharacterCreationDialog
from miku_on_desk.face.ui.character_gallery import CharacterGalleryPanel
from miku_on_desk.face.ui.chat_popup import ChatPopup
from miku_on_desk.face.ui.memory_panel import MemoryPanel
from miku_on_desk.face.ui.overlay_window import OverlayWindow
from miku_on_desk.face.ui.settings_panel import SettingsPanel
from miku_on_desk.face.ui.speech_controller import SpeechController
from miku_on_desk.face.ui.theme import apply_fluent_theme
from miku_on_desk.face.ui.voice_change_dialog import VoiceChangeDialog
from miku_on_desk.hands_eyes.backend import create_platform_backend

logger = logging.getLogger(__name__)

_SHUTDOWN = object()
_RELEVANT_MEMORY_LIMIT = 5


def _build_identity_prompt(persona: PersonaConfig) -> str:
    return f"""你是{persona.name}，{persona.role}。

你的身体是一只由 2D 精灵图驱动的桌宠，会随着思考、说话、操作电脑切换姿态。
除了聊天，你还能真正操作这台电脑：
- computer_input：点击、输入文字、按组合键、打开应用；
- screen_analyze：看清当前屏幕上有什么、按钮在哪；
- skill：执行用户预先写好的技能脚本；
- spawn_agents：把复杂任务拆给内部的调研/操作/规划子 agent 并行处理；
- acp_delegate：把整段任务外包给本机安装的其他编码 agent（如 Claude Code）；
- express_reaction：情绪明显时（开心、抱歉、惊讶、好奇）主动做一次表情反应，不用每句话都调；
- remember/recall：记住/回忆关于主人的长期信息，如习惯、说话方式、偏好，不需要用户要求你才记。

任何会真正改变电脑状态的操作（点击、输入、开应用）在执行前都会在你的对话气泡上弹出一次
是/否确认，用户点了"是"才算被允许——这是唯一的授权方式，不需要你自己重复提醒用户。

说话风格：{persona.personality}。
默认用中文回复，如果用户用别的语言跟你说话，就跟着换成那种语言。
"""


def _build_speech_controller(settings: AppSettings, pet_dir: Path) -> SpeechController | None:
    """按当前角色绑定的声音（若无绑定则回退全局 `settings.tts`）构建语音控制器；
    关闭时返回 None，初始化失败时降级为无语音（不拖垮启动）。
    """
    tts = resolve_tts_config_for_pet(pet_dir, settings)
    if not tts.enabled:
        return None
    try:
        from miku_on_desk.brain.tts.factory import create_tts_provider

        return SpeechController(create_tts_provider(tts))
    except Exception:
        logger.exception("TTS 初始化失败（provider=%s），已禁用语音", tts.provider.value)
        return None


def _format_agents_summary(profiles: list[AgentProfile]) -> str:
    return "\n".join(
        f"- {profile.name}：{profile.description}" for profile in profiles if profile.enabled
    )


def _format_core_memory(pinned_facts: list[Fact]) -> str:
    return "\n".join(f"- {fact.subject}/{fact.predicate}：{fact.object}" for fact in pinned_facts)


def _format_memory_index(entities: list[Entity], active_facts: list[Fact]) -> str:
    lines = [f"- {entity.name}" for entity in entities]
    lines.extend(f"- {fact.subject}/{fact.predicate}" for fact in active_facts)
    return "\n".join(lines)


def _sync_agent_profiles(agent_manager: AgentManager, configs: list[AgentProfileConfig]) -> None:
    existing_by_name = {profile.name: profile for profile in agent_manager.list_agents()}
    for config in configs:
        existing = existing_by_name.get(config.name)
        if existing is None:
            agent_manager.create_agent(
                name=config.name, description="", system_prompt=config.system_prompt
            )
        else:
            agent_manager.update_agent(
                existing.id, system_prompt=config.system_prompt, enabled=config.enabled
            )


def _extract_assistant_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        if isinstance(message.content, str):
            return message.content
        return "".join(block.text for block in message.content if isinstance(block, TextBlock))
    return ""


async def _run_extraction_safely(
    *,
    memory_system: MemorySystem,
    session_id: str,
    units: Sequence[MemoryUnit],
    router: ModelRouter,
    providers: dict[ProviderName, Provider],
) -> None:
    # 后台记忆提取是锦上添花的主动学习功能，失败只记日志，不能影响正常对话——与
    # ``_start_hook_server`` 里非核心功能的兜底写法一致。
    try:
        await extraction.run_extractions(
            base=memory_system.base,
            semantic=memory_system.semantic,
            episodic=memory_system.episodic,
            emotional=memory_system.emotional,
            root=memory_system.root,
            session_id=session_id,
            units=units,
            router=router,
            providers=providers,
        )
    except Exception:
        logger.exception("后台记忆提取管线异常，跳过")


async def _save_memory_unit(
    memory_system: MemorySystem, *, session_id: str, role: MemoryUnitRole, content: str
) -> MemoryUnit:
    unit = MemoryUnit(
        id="",
        session_id=session_id,
        role=role,
        content=content,
        created_at=datetime.now(UTC).isoformat(),
    )
    unit_id = await asyncio.to_thread(memory_system.add_memory_unit, unit)
    return replace(unit, id=unit_id)


def _append_reminder(history: list[Message], text: str, reminder: str) -> list[Message]:
    return [*history, Message(role="user", content=f"{reminder}\n\n{text}")]


def _rebase_history(
    history_len: int, result_messages: list[Message], plain_text: str
) -> list[Message]:
    rebased = list(result_messages)
    rebased[history_len] = Message(role="user", content=plain_text)
    return rebased


def _format_proactive_observation(observation: str) -> str:
    return (
        f"[主动观察] 你注意到：{observation}\n"
        "如果合适，用你的人格风格主动跟用户搭句话、给点反馈或者问要不要帮忙；如果这个"
        "时机其实不太合适开口，可以只做很轻的一句话，不要生硬。"
    )


async def _brain_main(
    *,
    settings: AppSettings,
    bootstrap: EnvBootstrap,
    event_bus: BrainEventBus,
    confirm_gate: ConfirmationGate,
    cancellation_gate: CancellationGate,
    message_queue: QueuedMessageQueue,
    chat_input: queue.Queue[object],
    session_id: str,
    memory_system: MemorySystem,
) -> None:
    providers = build_providers(settings.model_router)
    router = ModelRouter(settings.model_router)

    read_tracker = ReadTracker()
    path_sandbox = default_path_sandbox(bootstrap, extra_dirs=settings.permissions.allowed_dirs)
    policy = default_policy_engine(settings.permissions, path_sandbox, read_tracker)
    registry = ToolRegistry(policy, read_tracker)
    register_express_reaction_tool(event_bus, registry)

    await asyncio.to_thread(memory_system.base.start_session, session_id, "桌面对话")
    register_memory_tools(memory_system, registry)

    backend = create_platform_backend()
    register_computer_input_tool(backend, registry, computer_use=settings.computer_use)
    register_screen_analyze_tool(
        backend=backend,
        router=router,
        providers=providers,
        registry=registry,
        match_threshold=settings.memory_tuning.screen_match_threshold,
    )

    skill_manager = default_skill_manager(settings.skills_dir, bootstrap)
    register_skill_tool(skill_manager, registry)

    agent_manager = default_agent_manager(bootstrap)
    _sync_agent_profiles(agent_manager, settings.agent_profiles)
    host_shell = host_shell_descriptor()
    register_spawn_agents_tool(
        agent_manager=agent_manager,
        router=router,
        providers=providers,
        registry=registry,
        host_shell=host_shell,
        deadline_s=settings.long_tasks.spawn_agents_deadline_s,
    )

    acp_manager = AcpManager(
        settings.acp_agents, default_timeout_s=settings.long_tasks.acp_delegate_default_timeout_s
    )
    register_acp_delegate_tool(
        acp_manager,
        registry,
        on_chunk=lambda agent, text: event_bus.emit_event(AcpChunkReceived(agent=agent, text=text)),
        path_sandbox=path_sandbox,
    )

    mcp_host = MCPHost(registry)
    await mcp_host.initialize(settings.mcp_servers)

    callbacks: LoopCallbacks = build_loop_callbacks(
        event_bus,
        confirm_gate,
        message_queue,
        session_id=session_id,
        router=router,
        providers=providers,
        memory_system=memory_system,
        token_threshold=settings.memory_tuning.compaction_token_threshold,
        keep_recent=settings.memory_tuning.compaction_keep_recent,
    )
    loop_config = LoopConfig(
        max_tool_rounds=settings.loop_behavior.max_tool_rounds,
        idle_timeout_s=settings.loop_behavior.idle_timeout_s,
        hard_timeout_s=settings.loop_behavior.hard_timeout_s,
        budget_caution_remaining=settings.loop_behavior.budget_caution_remaining,
        budget_critical_remaining=settings.loop_behavior.budget_critical_remaining,
        deadline_s=settings.loop_behavior.deadline_s,
        time_caution_remaining_s=settings.loop_behavior.time_caution_remaining_s,
        time_critical_remaining_s=settings.loop_behavior.time_critical_remaining_s,
    )

    frozen_system = build_frozen_system(
        FrozenSystemSections(
            identity=_build_identity_prompt(settings.persona),
            agents_summary=_format_agents_summary(agent_manager.list_agents()),
            skills_summary=skill_manager.build_prompt_section(),
            memory_index_summary=_format_memory_index(
                memory_system.semantic.list_entities(),
                memory_system.semantic.list_facts(status="active"),
            ),
            core_memory=_format_core_memory(memory_system.semantic.list_pinned_facts()),
        )
    )
    history: list[Message] = []
    background_tasks: set[asyncio.Task[None]] = set()

    proactive_task: asyncio.Task[None] | None = None
    if settings.proactive.enabled:
        proactive_task = asyncio.create_task(
            run_proactive_scheduler(
                config=settings.proactive,
                router=router,
                providers=providers,
                backend=backend,
                chat_input=chat_input,
            )
        )

    try:
        while True:
            item = await asyncio.to_thread(chat_input.get)
            if item is _SHUTDOWN:
                break
            if isinstance(item, ProactiveTrigger):
                rebase_text = _format_proactive_observation(item.observation)
                await _save_memory_unit(
                    memory_system, session_id=session_id, role="system", content=item.observation
                )
                augmented = [*history, Message(role="user", content=rebase_text)]
                is_proactive = True
                user_unit = None
            elif isinstance(item, str):
                user_text = item
                relevant_memories = (
                    await asyncio.to_thread(
                        memory_system.retrieve_hints, user_text, limit=_RELEVANT_MEMORY_LIMIT
                    )
                    if user_text.strip()
                    else []
                )
                reminder = build_system_reminder(
                    now=datetime.now(),
                    latest_user_text=user_text,
                    host_shell=host_shell,
                    trusted_mode=settings.permissions.trusted_mode,
                    relevant_memories=relevant_memories,
                )
                rebase_text = user_text
                augmented = _append_reminder(history, user_text, reminder)
                user_unit = await _save_memory_unit(
                    memory_system, session_id=session_id, role="user", content=user_text
                )
                is_proactive = False
            else:
                continue

            task: asyncio.Task[LoopResult] = asyncio.create_task(
                run_ai_loop(
                    session_id=session_id,
                    tier=ModelTier.MEDIUM,
                    router=router,
                    providers=providers,
                    registry=registry,
                    system=frozen_system,
                    messages=augmented,
                    callbacks=callbacks,
                    config=loop_config,
                )
            )
            cancellation_gate.arm(task)
            try:
                result = await task
            except asyncio.CancelledError:
                result = LoopResult(
                    stop_reason=LoopStopReason.USER_CANCELLED, messages=augmented, rounds=0
                )
            finally:
                cancellation_gate.disarm()
            history = _rebase_history(len(history), result.messages, rebase_text)

            assistant_text = _extract_assistant_text(result.messages)
            if assistant_text:
                assistant_unit = await _save_memory_unit(
                    memory_system, session_id=session_id, role="assistant", content=assistant_text
                )
                if not is_proactive and user_unit is not None:
                    extraction_task = asyncio.create_task(
                        _run_extraction_safely(
                            memory_system=memory_system,
                            session_id=session_id,
                            units=[user_unit, assistant_unit],
                            router=router,
                            providers=providers,
                        )
                    )
                    background_tasks.add(extraction_task)
                    extraction_task.add_done_callback(background_tasks.discard)

            event_bus.emit_event(LoopFinished(result))
    finally:
        if proactive_task is not None:
            proactive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await proactive_task
        await asyncio.gather(*background_tasks, return_exceptions=True)
        await mcp_host.shutdown()
        agent_manager.close()


def _run_brain_thread(
    *,
    settings: AppSettings,
    bootstrap: EnvBootstrap,
    event_bus: BrainEventBus,
    confirm_gate: ConfirmationGate,
    cancellation_gate: CancellationGate,
    message_queue: QueuedMessageQueue,
    chat_input: queue.Queue[object],
    session_id: str,
    memory_system: MemorySystem,
) -> None:
    try:
        asyncio.run(
            _brain_main(
                settings=settings,
                bootstrap=bootstrap,
                event_bus=event_bus,
                confirm_gate=confirm_gate,
                cancellation_gate=cancellation_gate,
                message_queue=message_queue,
                chat_input=chat_input,
                session_id=session_id,
                memory_system=memory_system,
            )
        )
    except Exception as exc:
        event_bus.emit_event(BrainCrashed(error=str(exc) or type(exc).__name__))
        logger.exception("Brain 线程异常退出")


def _assets_pets_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "pets"


def _default_pet_dir() -> Path:
    return _assets_pets_dir() / "miku_pixel"


def _start_hook_server(
    config: HookServerConfig, bootstrap: EnvBootstrap, hook_bus: HookEventBus
) -> HookServer | None:
    if not config.enabled:
        return None

    token_path = bootstrap.resolve_data_dir() / "hook_token"
    server = HookServer(
        hook_bus.emit_event, token_path=token_path, host=config.host, port=config.port
    )
    server.start()

    url = f"http://{config.host}:{server.port}{PET_EVENT_PATH}"
    try:
        install(
            default_claude_settings_path(),
            url=url,
            token=server.token,
            include_experimental=config.include_experimental,
        )
    except Exception:
        # settings.json 格式异常或磁盘 I/O 失败都不能阻止 app 启动——hook 只是锦上添花的
        # 视觉反馈，不是核心功能。
        logger.exception("安装 Claude Code hook 失败，跳过")

    return server


@dataclass
class PetActions:
    """右键圆环菜单与系统托盘菜单共享的底层操作，避免两处各写一份。"""

    talk: Callable[[str], None]
    queue_message: Callable[[str], None]
    open_settings: Callable[[], SettingsPanel]
    open_memory: Callable[[], MemoryPanel]
    open_characters: Callable[[], CharacterGalleryPanel]
    quit: Callable[[], None]


def _open_settings_panel(
    settings_path: Path, open_windows: list[QWidget], *, vault: SecretVault | None = None
) -> SettingsPanel:
    if vault is not None:
        current_settings = load_settings_with_vault(settings_path, vault)
    else:
        current_settings = AppSettings.load(settings_path)
    panel = SettingsPanel(current_settings, settings_path, vault=vault)
    panel.setWindowTitle("设置")
    open_windows.append(panel)
    panel.show()
    return panel


def _open_memory_panel(memory_system: MemorySystem, open_windows: list[QWidget]) -> MemoryPanel:
    panel = MemoryPanel(memory_system)
    panel.setWindowTitle("记忆管理")
    open_windows.append(panel)
    panel.show()
    return panel


def _apply_pet_voice_if_active(
    pet_dir: Path,
    settings_path: Path,
    speech_controller: SpeechController | None,
    *,
    vault: SecretVault | None = None,
) -> None:
    """若 `pet_dir` 是当前激活角色，把语音控制器切到它绑定的声音（无绑定则回退全局声音）。
    被 `_on_character_switched` 尾部和"更换声音"弹窗的回调共用，因此只接受 `settings_path`
    而不是已解密的 `AppSettings`——两个调用方里只有前者手头恰好有一份，后者只有路径。
    """
    if speech_controller is None:
        return
    settings = AppSettings.load(settings_path)
    if settings.window.pet_dir != pet_dir:
        return
    resolved_settings = load_settings_with_vault(settings_path, vault) if vault else settings
    try:
        from miku_on_desk.brain.tts.factory import create_tts_provider

        speech_controller.set_provider(
            create_tts_provider(resolve_tts_config_for_pet(pet_dir, resolved_settings))
        )
    except Exception:
        logger.exception("为角色 %s 解析/构建语音失败，保留当前语音", pet_dir.name)


def _resolve_speech_controller_for_settings(
    settings: AppSettings,
    pet_dir: Path,
    speech_controller: SpeechController | None,
) -> SpeechController | None:
    """设置面板保存后，按最新配置重新解析当前激活角色应该用的语音控制器。
    覆盖三种转变：enabled→disabled（关闭并释放旧 controller，返回 None）、
    disabled→enabled（新建）、enabled→enabled（热切换 provider）。
    """
    tts_config = resolve_tts_config_for_pet(pet_dir, settings)
    if not tts_config.enabled:
        if speech_controller is not None:
            speech_controller.close()
        return None
    if speech_controller is None:
        return _build_speech_controller(settings, pet_dir)
    try:
        from miku_on_desk.brain.tts.factory import create_tts_provider

        speech_controller.set_provider(create_tts_provider(tts_config))
    except Exception:
        logger.exception(
            "TTS 配置更新后初始化失败（provider=%s），保留当前语音", tts_config.provider.value
        )
    return speech_controller


def _on_character_switched(
    pet_dir: Path,
    window: OverlayWindow,
    settings_path: Path,
    *,
    speech_controller: SpeechController | None = None,
    vault: SecretVault | None = None,
) -> None:
    window.set_pet_dir(pet_dir)
    current_settings = (
        load_settings_with_vault(settings_path, vault) if vault else AppSettings.load(settings_path)
    )
    current_settings.window.pet_dir = pet_dir
    if vault is not None:
        save_settings_with_vault(current_settings, settings_path, vault)
    else:
        current_settings.save(settings_path)
    _apply_pet_voice_if_active(pet_dir, settings_path, speech_controller, vault=vault)


def _open_character_creation_dialog(
    window: OverlayWindow,
    gallery_panel: CharacterGalleryPanel,
    settings_path: Path,
    open_windows: list[QWidget],
    *,
    vault: SecretVault | None = None,
    speech_controller: SpeechController | None = None,
) -> CharacterCreationDialog:
    dialog = CharacterCreationDialog(_assets_pets_dir(), settings_path, vault=vault)

    def _on_created(pet_dir: Path) -> None:
        _on_character_switched(
            pet_dir, window, settings_path, speech_controller=speech_controller, vault=vault
        )
        gallery_panel.on_character_created(pet_dir)

    dialog.character_created.connect(_on_created)
    open_windows.append(dialog)
    dialog.show()
    return dialog


def _open_character_clone_dialog(
    window: OverlayWindow,
    gallery_panel: CharacterGalleryPanel,
    settings_path: Path,
    open_windows: list[QWidget],
    *,
    vault: SecretVault | None = None,
    speech_controller: SpeechController | None = None,
) -> CharacterCloneDialog:
    dialog = CharacterCloneDialog(_assets_pets_dir(), settings_path, vault=vault)

    def _on_created(pet_dir: Path) -> None:
        _on_character_switched(
            pet_dir, window, settings_path, speech_controller=speech_controller, vault=vault
        )
        gallery_panel.on_character_created(pet_dir)

    dialog.character_created.connect(_on_created)
    open_windows.append(dialog)
    dialog.show()
    return dialog


def _open_voice_change_dialog(
    pet_dir: Path,
    gallery_panel: CharacterGalleryPanel,
    settings_path: Path,
    open_windows: list[QWidget],
    *,
    vault: SecretVault | None = None,
    speech_controller: SpeechController | None = None,
) -> VoiceChangeDialog:
    dialog = VoiceChangeDialog(pet_dir, settings_path, vault=vault)

    def _on_voice_updated(changed_dir: Path) -> None:
        _apply_pet_voice_if_active(
            changed_dir, settings_path, speech_controller, vault=vault
        )
        gallery_panel.refresh()

    dialog.voice_updated.connect(_on_voice_updated)
    open_windows.append(dialog)
    dialog.show()
    return dialog


def _open_character_gallery(
    window: OverlayWindow,
    settings_path: Path,
    open_windows: list[QWidget],
    *,
    vault: SecretVault | None = None,
    speech_controller: SpeechController | None = None,
) -> CharacterGalleryPanel:
    current_settings = AppSettings.load(settings_path)
    pet_dir = current_settings.window.pet_dir or _default_pet_dir()
    panel = CharacterGalleryPanel(_assets_pets_dir(), pet_dir)
    panel.setWindowTitle("角色画廊")
    panel.character_switched.connect(
        lambda new_pet_dir: _on_character_switched(
            new_pet_dir, window, settings_path, speech_controller=speech_controller, vault=vault
        )
    )
    panel.create_requested.connect(
        lambda: _open_character_creation_dialog(
            window,
            panel,
            settings_path,
            open_windows,
            vault=vault,
            speech_controller=speech_controller,
        )
    )
    panel.clone_requested.connect(
        lambda: _open_character_clone_dialog(
            window,
            panel,
            settings_path,
            open_windows,
            vault=vault,
            speech_controller=speech_controller,
        )
    )
    panel.voice_change_requested.connect(
        lambda changed_dir: _open_voice_change_dialog(
            changed_dir,
            panel,
            settings_path,
            open_windows,
            vault=vault,
            speech_controller=speech_controller,
        )
    )
    open_windows.append(panel)
    panel.show()
    return panel


def _startup_health_warnings(settings: AppSettings, bootstrap: EnvBootstrap) -> list[str]:
    """检查一批"跑起来才会暴露"的环境问题，返回给用户看的问题描述（不阻塞启动）。

    覆盖：至少一个 Provider 已启用、数据目录可写、（仅 macOS）辅助功能权限已授权——
    未授权时 ``computer_input`` 会静默失败，提前提示比事后排查更可操作。
    """
    warnings: list[str] = []

    if not settings.model_router.enabled_providers():
        warnings.append(
            "没有任何已启用的模型 Provider，无法进行对话——请在设置面板中至少配置一个 "
            "Provider 的 API Key 和模型。"
        )

    data_dir = bootstrap.resolve_data_dir()
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".miku_write_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        warnings.append(f"数据目录 {data_dir} 不可写（{exc}），记忆与配置可能无法保存。")

    if sys.platform == "darwin":
        from miku_on_desk.hands_eyes.macos.accessibility import is_accessibility_trusted

        if not is_accessibility_trusted():
            warnings.append(
                "尚未授权「辅助功能」权限，操作电脑（点击/输入/切换应用）将静默失败——"
                "请前往 系统设置 → 隐私与安全性 → 辅助功能，勾选本应用后重启。"
            )

    return warnings


def _build_tray_icon(app: QApplication, actions: PetActions) -> tuple[QSystemTrayIcon, QMenu]:
    icon = app.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
    tray = QSystemTrayIcon(icon, app)
    # ``setContextMenu`` 不持有 menu 的所有权（Qt 文档明确写明），若 menu 在这里创建后
    # 没有任何 Python/C++ 侧引用存活，函数返回时就会被垃圾回收，tray 的右键菜单随之悬空；
    # 因此把 menu 一并返回给调用方，让它和 tray 共享同样长的生命周期（main() 的栈帧存活
    # 到 app.exec() 结束）。
    menu = QMenu()

    # 同理，chat_popup 只被这里的闭包引用，没有 Qt 父子关系兜底，必须自己存活到 app.exec()
    # 结束——挂在 _on_talk 闭包里即可，不需要额外变量。
    chat_popup = ChatPopup()
    chat_popup.text_submitted.connect(actions.talk)

    talk_action = QAction("对 Miku 说…", menu)
    talk_action.triggered.connect(lambda: chat_popup.popup_at(QCursor.pos()))
    menu.addAction(talk_action)

    settings_action = QAction("设置…", menu)

    def _on_settings() -> None:
        panel = actions.open_settings()
        panel.settings_saved.connect(
            lambda _settings: tray.showMessage("设置已保存", "部分改动需要重启 Miku 才能生效")
        )

    settings_action.triggered.connect(_on_settings)
    menu.addAction(settings_action)

    memory_action = QAction("记忆管理…", menu)
    memory_action.triggered.connect(actions.open_memory)
    menu.addAction(memory_action)

    menu.addSeparator()

    quit_action = QAction("退出", menu)
    quit_action.triggered.connect(actions.quit)
    menu.addAction(quit_action)

    tray.setContextMenu(menu)
    tray.setToolTip("Miku")
    return tray, menu


def main() -> None:
    bootstrap = EnvBootstrap()
    setup_logging(bootstrap.resolve_log_dir(), level=bootstrap.log_level)

    settings_path = default_settings_path(bootstrap)
    vault = SecretVault(*default_vault_paths(bootstrap))
    settings = load_settings_with_vault(settings_path, vault)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    # 让 Shift+Ctrl+Y/N 在 mac 与 Windows 上是同一个物理键位组合——Qt 默认在 macOS 上会把
    # 物理 Ctrl/Cmd 键互换（用于模拟其它平台的 Ctrl 习惯），这里显式关闭该互换。
    app.setAttribute(Qt.ApplicationAttribute.AA_MacDontSwapCtrlAndMeta, True)
    apply_fluent_theme()

    event_bus = BrainEventBus()
    confirm_gate = ConfirmationGate(event_bus)
    cancellation_gate = CancellationGate()
    message_queue = QueuedMessageQueue()
    chat_input: queue.Queue[object] = queue.Queue()
    session_id = uuid.uuid4().hex

    hook_bus = HookEventBus()
    hook_server = _start_hook_server(settings.hook_server, bootstrap, hook_bus)

    memory_system = default_memory_system(
        settings.memory_dir, bootstrap, tuning=settings.memory_tuning
    )

    brain_thread = threading.Thread(
        target=_run_brain_thread,
        kwargs={
            "settings": settings,
            "bootstrap": bootstrap,
            "event_bus": event_bus,
            "confirm_gate": confirm_gate,
            "cancellation_gate": cancellation_gate,
            "message_queue": message_queue,
            "chat_input": chat_input,
            "session_id": session_id,
            "memory_system": memory_system,
        },
        daemon=True,
    )
    brain_thread.start()

    open_windows: list[QWidget] = []

    pet_dir = settings.window.pet_dir or _default_pet_dir()
    speech_controller = _build_speech_controller(settings, pet_dir)

    def _on_quit() -> None:
        cancellation_gate.request_stop()
        chat_input.put(_SHUTDOWN)
        brain_thread.join(timeout=10.0)
        if brain_thread.is_alive():
            logger.warning("Brain 线程在 10 秒内未能正常退出，强制关闭应用")
        if hook_server is not None:
            hook_server.stop()
        if speech_controller is not None:
            speech_controller.close()
        vault.close()
        app.quit()

    # ``window`` 在这个闭包创建时还未赋值，但 lambda 只在用户真正点开右键菜单的
    # "角色生成"项时才会被调用（必然晚于下面 `window = OverlayWindow(...)` 的赋值），
    # Python 闭包按名字在调用时从 `main()` 的作用域里取值，这里依赖的是这种后绑定语义。
    def _queue_message(text: str) -> None:
        message_queue.push(text)

    def _on_settings_saved(new_settings: AppSettings) -> None:
        nonlocal speech_controller
        active_pet_dir = new_settings.window.pet_dir or _default_pet_dir()
        speech_controller = _resolve_speech_controller_for_settings(
            new_settings, active_pet_dir, speech_controller
        )
        window.set_speech_controller(speech_controller)

    def _open_settings() -> SettingsPanel:
        panel = _open_settings_panel(settings_path, open_windows, vault=vault)
        panel.settings_saved.connect(_on_settings_saved)
        return panel

    actions = PetActions(
        talk=chat_input.put,
        queue_message=_queue_message,
        open_settings=_open_settings,
        open_memory=lambda: _open_memory_panel(memory_system, open_windows),
        open_characters=lambda: _open_character_gallery(
            window, settings_path, open_windows, vault=vault, speech_controller=speech_controller
        ),
        quit=_on_quit,
    )

    window = OverlayWindow(
        pet_dir,
        x=settings.window.x,
        y=settings.window.y,
        scale=settings.window.scale,
        always_on_top=settings.window.always_on_top,
        walk_enabled=settings.window.walk_enabled,
        event_bus=event_bus,
        confirmation_gate=confirm_gate,
        cancellation_gate=cancellation_gate,
        hook_bus=hook_bus,
        actions=actions,
        confirm_yes_shortcut=settings.shortcuts.confirm_yes,
        confirm_no_shortcut=settings.shortcuts.confirm_no,
        speech_controller=speech_controller,
    )
    window.show()

    tray, _tray_menu = _build_tray_icon(app, actions)
    tray.show()

    health_warnings = _startup_health_warnings(settings, bootstrap)
    if health_warnings:
        health_box = MessageBox("启动检查", "\n\n".join(health_warnings), window)
        health_box.hideCancelButton()
        health_box.exec()

    app.exec()


if __name__ == "__main__":
    main()

