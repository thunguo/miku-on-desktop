"""main.py 里纯函数式装配逻辑的回归测试：provider 构造、prompt 片段格式化、
agent profile 同步、历史消息 rebase。``main()``/``_brain_main`` 本身装配 Qt 与
asyncio 事件循环，不在这里测试。
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
from PySide6.QtWidgets import QApplication, QWidget

from miku_on_desk.brain.agents.manager import AgentManager, AgentProfile
from miku_on_desk.brain.memory.models import Entity, Fact
from miku_on_desk.brain.memory.system import default_memory_system
from miku_on_desk.brain.providers.anthropic_provider import AnthropicProvider
from miku_on_desk.brain.providers.base import Message, TextBlock, ToolUseBlock
from miku_on_desk.brain.providers.gemini_provider import GeminiProvider
from miku_on_desk.brain.providers.openai_compatible_provider import OpenAICompatibleProvider
from miku_on_desk.bridge.events import (
    BrainCrashed,
    BrainEventBus,
    CancellationGate,
    ConfirmationGate,
    QueuedMessageQueue,
)
from miku_on_desk.config.settings import (
    AgentProfileConfig,
    AppSettings,
    EnvBootstrap,
    ModelRouterConfig,
    ModelTier,
    PersonaConfig,
    ProviderConfig,
    ProviderName,
)
from miku_on_desk.face.ui.character_gallery import CharacterGalleryPanel
from miku_on_desk.face.ui.overlay_window import OverlayWindow
from miku_on_desk.main import (
    _append_reminder,
    _build_identity_prompt,
    _build_providers,
    _extract_assistant_text,
    _format_agents_summary,
    _format_core_memory,
    _format_memory_index,
    _open_character_creation_dialog,
    _open_memory_panel,
    _rebase_history,
    _run_brain_thread,
    _startup_health_warnings,
    _sync_agent_profiles,
)


def test_build_providers_only_constructs_enabled_providers() -> None:
    config = ModelRouterConfig(
        anthropic=ProviderConfig(api_key="sk-ant", models={ModelTier.FAST: "haiku"}),
        openai=ProviderConfig(api_key=None, models={}),
        gemini=ProviderConfig(api_key="sk-gemini", models={ModelTier.HEAVY: "gemini-pro"}),
    )

    providers = _build_providers(config)

    assert set(providers) == {ProviderName.ANTHROPIC, ProviderName.GEMINI}
    assert isinstance(providers[ProviderName.ANTHROPIC], AnthropicProvider)
    assert isinstance(providers[ProviderName.GEMINI], GeminiProvider)


def test_build_providers_constructs_openai_compatible_provider() -> None:
    config = ModelRouterConfig(
        openai=ProviderConfig(api_key="sk-openai", models={ModelTier.MEDIUM: "gpt-5"})
    )

    providers = _build_providers(config)

    assert isinstance(providers[ProviderName.OPENAI], OpenAICompatibleProvider)


def test_format_agents_summary_skips_disabled_profiles() -> None:
    profiles = [
        AgentProfile(id="1", name="researcher", description="调研", system_prompt="x"),
        AgentProfile(id="2", name="ghost", description="已禁用", system_prompt="x", enabled=False),
    ]

    summary = _format_agents_summary(profiles)

    assert "researcher：调研" in summary
    assert "ghost" not in summary


def _make_fact(*, subject: str, predicate: str, value: str, pinned: bool = False) -> Fact:
    return Fact(
        id="",
        subject=subject,
        subject_type="person",
        predicate=predicate,
        object=value,
        object_type="concept",
        confidence=1.0,
        source=[],
        valid_from="2026-01-01T00:00:00",
        recorded_at="2026-01-01T00:00:00",
        extracted_by="tool:remember",
        status="active",
        pinned=pinned,
    )


def test_format_core_memory_renders_key_value_lines() -> None:
    facts = [
        _make_fact(subject="user", predicate="name", value="tew", pinned=True),
        _make_fact(subject="user", predicate="role", value="engineer", pinned=True),
    ]

    formatted = _format_core_memory(facts)

    assert formatted == "- user/name：tew\n- user/role：engineer"


def test_format_memory_index_renders_keys_only() -> None:
    entities = [Entity(id="e1", name="tew", type="person")]
    facts = [
        _make_fact(subject="user", predicate="habits/sleep_schedule", value="喜欢熬夜"),
        _make_fact(subject="user", predicate="habits/coffee", value="喝美式"),
    ]

    formatted = _format_memory_index(entities, facts)

    assert formatted == "- tew\n- user/habits/sleep_schedule\n- user/habits/coffee"
    assert "喜欢熬夜" not in formatted


def test_build_identity_prompt_interpolates_persona_fields() -> None:
    persona = PersonaConfig(name="小明", role="程序员助手", personality="冷静克制")

    prompt = _build_identity_prompt(persona)

    assert "你是小明，程序员助手。" in prompt
    assert "说话风格：冷静克制。" in prompt
    assert "remember/recall" in prompt
    assert "3D 模型" not in prompt
    assert "2D 精灵图" in prompt


def test_sync_agent_profiles_creates_new_profile(tmp_path: Path) -> None:
    manager = AgentManager(tmp_path / "agents.db")
    try:
        _sync_agent_profiles(
            manager, [AgentProfileConfig(name="custom", system_prompt="你是自定义助手")]
        )

        created = next(p for p in manager.list_agents() if p.name == "custom")
        assert created.system_prompt == "你是自定义助手"
        assert created.builtin is False
    finally:
        manager.close()


def test_sync_agent_profiles_updates_builtin_without_renaming(tmp_path: Path) -> None:
    manager = AgentManager(tmp_path / "agents.db")
    try:
        builtin = next(p for p in manager.list_agents() if p.builtin)

        _sync_agent_profiles(
            manager,
            [AgentProfileConfig(name=builtin.name, system_prompt="更新后的提示词", enabled=False)],
        )

        updated = manager.get_agent(builtin.id)
        assert updated is not None
        assert updated.name == builtin.name
        assert updated.system_prompt == "更新后的提示词"
        assert updated.enabled is False
    finally:
        manager.close()


def test_extract_assistant_text_returns_last_assistant_message_plain_string() -> None:
    history = [
        Message(role="user", content="你好"),
        Message(role="assistant", content="你好呀"),
        Message(role="user", content="在干嘛"),
        Message(role="assistant", content="在摸鱼"),
    ]

    assert _extract_assistant_text(history) == "在摸鱼"


def test_extract_assistant_text_joins_text_blocks_and_skips_tool_use() -> None:
    history = [
        Message(
            role="assistant",
            content=[
                TextBlock(text="让我看看"),
                ToolUseBlock(id="1", name="screen_analyze", input={}),
                TextBlock(text="好了"),
            ],
        )
    ]

    assert _extract_assistant_text(history) == "让我看看好了"


def test_extract_assistant_text_returns_empty_string_when_no_assistant_message() -> None:
    assert _extract_assistant_text([Message(role="user", content="你好")]) == ""


def test_append_reminder_prefixes_latest_user_text_with_reminder() -> None:
    history = [Message(role="assistant", content="早")]

    result = _append_reminder(history, "帮我开一下计算器", "<reminder>现在是早上</reminder>")

    assert len(result) == 2
    assert result[0] is history[0]
    assert result[1].role == "user"
    assert result[1].content == "<reminder>现在是早上</reminder>\n\n帮我开一下计算器"


def test_rebase_history_replaces_augmented_turn_with_plain_text() -> None:
    history = [Message(role="assistant", content="早")]
    result_messages = [
        *history,
        Message(role="user", content="<reminder>...</reminder>\n\n帮我开一下计算器"),
        Message(role="assistant", content="好的"),
    ]

    rebased = _rebase_history(len(history), result_messages, "帮我开一下计算器")

    assert rebased[1] == Message(role="user", content="帮我开一下计算器")
    assert rebased[2] == Message(role="assistant", content="好的")


_FRAME_SIZE = 4


def _make_pet_dir(base: Path, name: str) -> Path:
    pet_dir = base / name
    pet_dir.mkdir()
    Image.new("RGBA", (_FRAME_SIZE, _FRAME_SIZE), (255, 0, 0, 255)).save(
        pet_dir / "spritesheet.png"
    )
    meta = {
        "pet_name": name,
        "frame_width": _FRAME_SIZE,
        "frame_height": _FRAME_SIZE,
        "columns": 1,
        "rows": 1,
        "fallback_state": "idle",
        "states": {"idle": {"row": 0, "frame_count": 1, "fps": 1.0, "loop": True}},
    }
    (pet_dir / "pet.json").write_text(json.dumps(meta), encoding="utf-8")
    return pet_dir


def test_character_created_hot_switches_window_and_persists_settings(
    qapp: QApplication, tmp_path: Path
) -> None:
    """新角色生成完成后应立即热切换桌宠窗口并持久化 settings，不需要重启应用，
    也不需要用户再手动去画廊点一次切换。
    """
    assets_pets_dir = tmp_path / "assets_pets"
    assets_pets_dir.mkdir()
    old_pet_dir = _make_pet_dir(assets_pets_dir, "old_pet")
    new_pet_dir = _make_pet_dir(assets_pets_dir, "new_pet")
    settings_path = tmp_path / "settings.json"

    window = OverlayWindow(old_pet_dir)
    window.show()
    gallery_panel = CharacterGalleryPanel(assets_pets_dir, old_pet_dir)

    dialog = _open_character_creation_dialog(window, gallery_panel, settings_path, [])
    dialog.character_created.emit(new_pet_dir)

    assert window._meta.pet_name == "new_pet"
    assert window._sprite_widget.isVisibleTo(window) is True
    assert AppSettings.load(settings_path).window.pet_dir == new_pet_dir
    assert gallery_panel._current_pet_dir == new_pet_dir


def test_run_brain_thread_emits_brain_crashed_when_brain_main_raises(
    qapp: QApplication,
) -> None:
    bus = BrainEventBus()
    captured: list[object] = []
    bus.brain_event.connect(captured.append)

    with patch("miku_on_desk.main._brain_main", side_effect=RuntimeError("炸了")):
        thread = threading.Thread(
            target=_run_brain_thread,
            kwargs={
                "settings": Mock(),
                "bootstrap": Mock(),
                "event_bus": bus,
                "confirm_gate": ConfirmationGate(bus),
                "cancellation_gate": CancellationGate(),
                "message_queue": QueuedMessageQueue(),
                "chat_input": queue.Queue(),
                "session_id": "s1",
                "memory_system": Mock(),
            },
        )
        thread.start()
        thread.join(timeout=5.0)

    qapp.processEvents()

    assert captured == [BrainCrashed(error="炸了")]


def test_open_memory_panel_reuses_passed_in_memory_system_instance(
    qapp: QApplication, tmp_path: Path
) -> None:
    """阶段 E 单例化：`_open_memory_panel` 必须复用调用方传入的 `MemorySystem`，
    不能像之前那样自己重新构造一份——否则 UI 线程和 Brain 线程会各自持有互不相通的实例，
    同时写同一批存储文件时产生竞态。
    """
    memory_system = default_memory_system(tmp_path / "memory")
    open_windows: list[QWidget] = []

    panel = _open_memory_panel(memory_system, open_windows)

    assert panel._system is memory_system
    assert panel in open_windows


def _enabled_settings() -> AppSettings:
    settings = AppSettings()
    settings.model_router.anthropic = ProviderConfig(
        api_key="sk-ant", models={ModelTier.FAST: "haiku"}
    )
    return settings


def test_startup_health_warnings_flags_missing_enabled_provider(tmp_path: Path) -> None:
    bootstrap = EnvBootstrap(data_dir=tmp_path / "data")

    with patch(
        "miku_on_desk.hands_eyes.macos.accessibility.is_accessibility_trusted",
        return_value=True,
    ):
        warnings = _startup_health_warnings(AppSettings(), bootstrap)

    assert any("Provider" in warning for warning in warnings)


def test_startup_health_warnings_flags_unwritable_data_dir(tmp_path: Path) -> None:
    blocked_path = tmp_path / "blocked"
    blocked_path.write_text("", encoding="utf-8")
    bootstrap = EnvBootstrap(data_dir=blocked_path)

    with patch(
        "miku_on_desk.hands_eyes.macos.accessibility.is_accessibility_trusted",
        return_value=True,
    ):
        warnings = _startup_health_warnings(_enabled_settings(), bootstrap)

    assert any("不可写" in warning for warning in warnings)


def test_startup_health_warnings_flags_missing_accessibility_trust_on_macos(
    tmp_path: Path,
) -> None:
    bootstrap = EnvBootstrap(data_dir=tmp_path / "data")

    with (
        patch("sys.platform", "darwin"),
        patch(
            "miku_on_desk.hands_eyes.macos.accessibility.is_accessibility_trusted",
            return_value=False,
        ),
    ):
        warnings = _startup_health_warnings(_enabled_settings(), bootstrap)

    assert any("辅助功能" in warning for warning in warnings)


def test_startup_health_warnings_empty_when_everything_healthy(tmp_path: Path) -> None:
    bootstrap = EnvBootstrap(data_dir=tmp_path / "data")

    with patch(
        "miku_on_desk.hands_eyes.macos.accessibility.is_accessibility_trusted",
        return_value=True,
    ):
        warnings = _startup_health_warnings(_enabled_settings(), bootstrap)

    assert warnings == []

