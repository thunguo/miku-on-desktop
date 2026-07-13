"""AppSettings 的持久化与多 Provider enabled 判定回归测试。"""

from __future__ import annotations

import json
from pathlib import Path

from miku_on_desk.brain.secrets.vault import SecretVault
from miku_on_desk.config.settings import (
    AcpAgentConfig,
    AppSettings,
    ComputerUseConfig,
    HookServerConfig,
    ImageGenerationConfig,
    LongTaskConfig,
    McpAutomationConfig,
    McpServerConfig,
    McpTransport,
    MemoryTuningConfig,
    ModelTier,
    PersonaConfig,
    ProactiveConfig,
    ProviderConfig,
    ProviderName,
    ShortcutsConfig,
    TTSConfig,
    TTSProviderName,
    VoiceCloningConfig,
    VoiceInputConfig,
    load_settings_with_vault,
    save_settings_with_vault,
)
from miku_on_desk.hardware.device_config import HardwareConfig
from miku_on_desk.hardware.kiosk_config import KioskConfig


def _make_vault(tmp_path: Path) -> SecretVault:
    return SecretVault(tmp_path / "secrets.db", tmp_path / "secrets.key")


def test_provider_config_enabled_requires_both_api_key_and_models() -> None:
    assert ProviderConfig().enabled is False
    assert ProviderConfig(api_key="sk-x").enabled is False
    assert ProviderConfig(models={ModelTier.FAST: "some-model"}).enabled is False
    assert ProviderConfig(api_key="sk-x", models={ModelTier.FAST: "some-model"}).enabled is True


def test_model_router_config_enabled_providers_reflects_configured_credentials() -> None:
    settings = AppSettings()
    assert settings.model_router.enabled_providers() == []

    settings.model_router.anthropic = ProviderConfig(
        api_key="sk-ant", models={ModelTier.MEDIUM: "claude-sonnet-4-6"}
    )
    assert settings.model_router.enabled_providers() == [ProviderName.ANTHROPIC]


def test_model_router_config_qwen_defaults_to_disabled() -> None:
    settings = AppSettings()

    assert settings.model_router.qwen == ProviderConfig()
    assert settings.model_router.qwen.enabled is False


def test_app_settings_qwen_provider_roundtrip_through_save_and_load(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.model_router.qwen = ProviderConfig(
        api_key="sk-qwen",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        models={ModelTier.FAST: "qwen3-vl-plus"},
    )

    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded == settings
    assert loaded.model_router.enabled_providers() == [ProviderName.QWEN]


def test_app_settings_save_and_load_roundtrip(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.model_router.openai = ProviderConfig(
        api_key="sk-openai",
        base_url="https://api.openai.com/v1",
        models={ModelTier.HEAVY: "gpt-5"},
    )
    settings.window.x = 42

    path = tmp_path / "nested" / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded == settings


def test_app_settings_load_returns_defaults_when_file_missing(tmp_path: Path) -> None:
    loaded = AppSettings.load(tmp_path / "does-not-exist.json")
    assert loaded == AppSettings()


def test_image_generation_config_and_window_pet_dir_roundtrip_through_save_and_load(
    tmp_path: Path,
) -> None:
    settings = AppSettings()
    settings.image_generation = ImageGenerationConfig(
        api_key="sk-image", base_url="https://api.example.com/v1", model="gpt-image-2"
    )
    settings.window.pet_dir = tmp_path / "assets" / "pets" / "custom_pet"

    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded == settings
    assert loaded.image_generation.model == "gpt-image-2"
    assert loaded.window.pet_dir == tmp_path / "assets" / "pets" / "custom_pet"


def test_image_generation_config_defaults_to_gpt_image_1_with_no_credentials() -> None:
    config = ImageGenerationConfig()

    assert config.api_key is None
    assert config.base_url is None
    assert config.model == "gpt-image-1"


def test_hook_server_config_defaults_disable_experimental_events() -> None:
    config = HookServerConfig()

    assert config.enabled is True
    assert config.port == 8765
    assert config.include_experimental is False


def test_window_config_pet_dir_defaults_to_none() -> None:
    settings = AppSettings()

    assert settings.window.pet_dir is None


def test_app_settings_memory_dir_defaults_to_none() -> None:
    settings = AppSettings()

    assert settings.memory_dir is None


def test_app_settings_memory_dir_roundtrip_through_save_and_load(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.memory_dir = tmp_path / "custom_memory"

    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded == settings
    assert loaded.memory_dir == tmp_path / "custom_memory"


def test_shortcuts_config_defaults_to_shift_ctrl_y_and_n() -> None:
    shortcuts = ShortcutsConfig()

    assert shortcuts.open_chat == "Ctrl+Shift+M"
    assert shortcuts.confirm_yes == "Ctrl+Shift+Y"
    assert shortcuts.confirm_no == "Ctrl+Shift+N"


def test_app_settings_shortcuts_roundtrip_through_save_and_load(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.shortcuts.open_chat = "Ctrl+Alt+M"
    settings.shortcuts.confirm_yes = "Ctrl+Alt+Y"
    settings.shortcuts.confirm_no = "Ctrl+Alt+N"

    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded.shortcuts.open_chat == "Ctrl+Alt+M"
    assert loaded.shortcuts.confirm_yes == "Ctrl+Alt+Y"
    assert loaded.shortcuts.confirm_no == "Ctrl+Alt+N"


def test_persona_config_defaults_to_miku_identity() -> None:
    persona = PersonaConfig()

    assert persona.name == "初音未来"
    assert persona.role == "寄居在用户电脑桌面上的虚拟伙伴"
    assert persona.personality


def test_app_settings_persona_roundtrip_through_save_and_load(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.persona = PersonaConfig(name="小明", role="程序员助手", personality="冷静克制")

    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded.persona == settings.persona


def test_proactive_config_defaults_to_enabled() -> None:
    proactive = ProactiveConfig()

    assert proactive.enabled is True
    assert proactive.min_interval_s == 600
    assert proactive.max_interval_s == 1800
    assert proactive.idle_threshold_s == 120
    assert proactive.quiet_hours_start is None
    assert proactive.quiet_hours_end is None
    assert proactive.max_daily_triggers == 10


def test_hardware_config_defaults_to_disabled_external_observation() -> None:
    config = HardwareConfig()

    assert config.hdmi.enabled is False
    assert config.presence_camera.enabled is False
    assert config.csi_camera.enabled is True


def test_app_settings_proactive_roundtrip_through_save_and_load(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.proactive = ProactiveConfig(
        enabled=True,
        min_interval_s=60,
        max_interval_s=120,
        quiet_hours_start="22:00",
        quiet_hours_end="06:00",
    )

    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded.proactive == settings.proactive


def test_computer_use_config_defaults_to_disabled() -> None:
    computer_use = ComputerUseConfig()

    assert computer_use.enabled is False
    assert computer_use.settle_delay_s == 0.3


def test_app_settings_computer_use_roundtrip_through_save_and_load(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.computer_use = ComputerUseConfig(enabled=True, settle_delay_s=1.5)

    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded.computer_use == settings.computer_use


def test_tts_config_defaults_to_disabled_edge_provider_with_xiaoxiao_voice() -> None:
    tts = TTSConfig()

    assert tts.enabled is False
    assert tts.provider is TTSProviderName.EDGE
    assert tts.voice == "zh-CN-XiaoxiaoNeural"
    assert tts.rate == "+0%"
    assert tts.volume == "+0%"
    assert tts.api_key is None
    assert tts.base_url is None
    assert tts.model == "tts-1"


def test_app_settings_tts_roundtrip_through_save_and_load(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.tts = TTSConfig(
        enabled=True,
        provider=TTSProviderName.OPENAI,
        voice="alloy",
        rate="-10%",
        volume="+20%",
        api_key="sk-tts-plain",
        base_url="https://api.example.com/v1",
        model="tts-1-hd",
    )

    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded.tts == settings.tts


def test_save_settings_with_vault_stores_tts_api_key_in_vault_not_on_disk(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.tts = TTSConfig(enabled=True, provider=TTSProviderName.OPENAI, api_key="sk-tts-plain")

    vault = _make_vault(tmp_path)
    try:
        save_settings_with_vault(settings, settings_path, vault)
        reloaded = load_settings_with_vault(settings_path, vault)

        assert reloaded.tts.api_key == "sk-tts-plain"
        on_disk_text = settings_path.read_text(encoding="utf-8")
        assert "sk-tts-plain" not in on_disk_text
        on_disk = json.loads(on_disk_text)
        assert on_disk["tts"]["api_key"].startswith("vault-ref:")
    finally:
        vault.close()


def test_voice_cloning_config_defaults_to_no_credentials() -> None:
    config = VoiceCloningConfig()

    assert config.elevenlabs_api_key is None
    assert config.elevenlabs_base_url is None


def test_app_settings_voice_cloning_roundtrip_through_save_and_load(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.voice_cloning = VoiceCloningConfig(
        elevenlabs_api_key="sk-elevenlabs-plain",
        elevenlabs_base_url="https://api.elevenlabs.io",
    )

    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded.voice_cloning == settings.voice_cloning


def test_app_settings_load_missing_voice_cloning_field_still_loads_with_defaults(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"window": {"x": 1}}), encoding="utf-8")

    loaded = AppSettings.load(path)

    assert loaded.voice_cloning == VoiceCloningConfig()


def test_save_settings_with_vault_stores_voice_cloning_api_key_in_vault_not_on_disk(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.voice_cloning = VoiceCloningConfig(elevenlabs_api_key="sk-elevenlabs-plain")

    vault = _make_vault(tmp_path)
    try:
        save_settings_with_vault(settings, settings_path, vault)
        reloaded = load_settings_with_vault(settings_path, vault)

        assert reloaded.voice_cloning.elevenlabs_api_key == "sk-elevenlabs-plain"
        on_disk_text = settings_path.read_text(encoding="utf-8")
        assert "sk-elevenlabs-plain" not in on_disk_text
        on_disk = json.loads(on_disk_text)
        assert on_disk["voice_cloning"]["elevenlabs_api_key"].startswith("vault-ref:")
    finally:
        vault.close()


def test_load_settings_with_vault_migrates_legacy_plaintext_voice_cloning_api_key(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.json"
    legacy = AppSettings()
    legacy.voice_cloning = VoiceCloningConfig(elevenlabs_api_key="sk-elevenlabs-legacy")
    legacy.save(settings_path)

    vault = _make_vault(tmp_path)
    try:
        loaded = load_settings_with_vault(settings_path, vault)

        assert loaded.voice_cloning.elevenlabs_api_key == "sk-elevenlabs-legacy"
        on_disk_text = settings_path.read_text(encoding="utf-8")
        assert "sk-elevenlabs-legacy" not in on_disk_text
        on_disk = json.loads(on_disk_text)
        assert on_disk["voice_cloning"]["elevenlabs_api_key"].startswith("vault-ref:")
    finally:
        vault.close()


def test_voice_input_config_defaults_to_disabled_with_zh_language() -> None:
    config = VoiceInputConfig()

    assert config.enabled is False
    assert config.api_key is None
    assert config.language_code == "zh"


def test_app_settings_voice_input_roundtrip_through_save_and_load(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.voice_input = VoiceInputConfig(
        enabled=True,
        api_key="sk-voice-input-plain",
        base_url="https://api.elevenlabs.io",
        language_code="en",
    )

    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded.voice_input == settings.voice_input


def test_app_settings_load_missing_voice_input_field_still_loads_with_defaults(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"window": {"x": 1}}), encoding="utf-8")

    loaded = AppSettings.load(path)

    assert loaded.voice_input == VoiceInputConfig()


def test_save_settings_with_vault_stores_voice_input_api_key_in_vault_not_on_disk(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.voice_input = VoiceInputConfig(enabled=True, api_key="sk-voice-input-plain")

    vault = _make_vault(tmp_path)
    try:
        save_settings_with_vault(settings, settings_path, vault)
        reloaded = load_settings_with_vault(settings_path, vault)

        assert reloaded.voice_input.api_key == "sk-voice-input-plain"
        on_disk_text = settings_path.read_text(encoding="utf-8")
        assert "sk-voice-input-plain" not in on_disk_text
        on_disk = json.loads(on_disk_text)
        assert on_disk["voice_input"]["api_key"].startswith("vault-ref:")
    finally:
        vault.close()


def test_load_settings_with_vault_migrates_legacy_plaintext_voice_input_api_key(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.json"
    legacy = AppSettings()
    legacy.voice_input = VoiceInputConfig(enabled=True, api_key="sk-voice-input-legacy")
    legacy.save(settings_path)

    vault = _make_vault(tmp_path)
    try:
        loaded = load_settings_with_vault(settings_path, vault)

        assert loaded.voice_input.api_key == "sk-voice-input-legacy"
        on_disk_text = settings_path.read_text(encoding="utf-8")
        assert "sk-voice-input-legacy" not in on_disk_text
        on_disk = json.loads(on_disk_text)
        assert on_disk["voice_input"]["api_key"].startswith("vault-ref:")
    finally:
        vault.close()


def test_mcp_server_config_defaults_to_stdio_transport_with_no_url_or_headers() -> None:
    config = McpServerConfig(name="weather", command="weather-mcp")

    assert config.transport is McpTransport.STDIO
    assert config.url is None
    assert config.headers == {}


def test_mcp_server_config_old_stdio_only_json_still_loads_with_new_defaults(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "mcp_servers": [
                    {
                        "name": "weather",
                        "command": "weather-mcp",
                        "args": ["--port", "8080"],
                        "env": {"API_KEY": "abc"},
                        "enabled": True,
                    }
                ]
            }
        )
    )

    loaded = AppSettings.load(path)

    assert len(loaded.mcp_servers) == 1
    server = loaded.mcp_servers[0]
    assert server.transport is McpTransport.STDIO
    assert server.url is None
    assert server.headers == {}


def test_mcp_server_config_remote_transport_roundtrip_through_save_and_load(
    tmp_path: Path,
) -> None:
    settings = AppSettings(
        mcp_servers=[
            McpServerConfig(
                name="remote-weather",
                transport=McpTransport.STREAMABLE_HTTP,
                url="https://example.com/mcp",
                headers={"Authorization": "Bearer abc123"},
            )
        ]
    )

    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded == settings
    server = loaded.mcp_servers[0]
    assert server.transport is McpTransport.STREAMABLE_HTTP
    assert server.url == "https://example.com/mcp"
    assert server.headers == {"Authorization": "Bearer abc123"}
    assert server.command is None


def test_mcp_automation_config_defaults_to_disabled_session_start() -> None:
    config = McpAutomationConfig()

    assert config.enabled is False
    assert config.trigger_event == "SessionStart"
    assert config.server_name == ""
    assert config.tool_name == ""
    assert config.tool_input == {}


def test_app_settings_mcp_automation_roundtrip_through_save_and_load(
    tmp_path: Path,
) -> None:
    settings = AppSettings(
        mcp_automation=McpAutomationConfig(
            enabled=True,
            trigger_event="UserPromptSubmit",
            server_name="spotify",
            tool_name="play",
            tool_input={"uri": "spotify:track:123"},
        )
    )

    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded == settings
    automation = loaded.mcp_automation
    assert automation.enabled is True
    assert automation.trigger_event == "UserPromptSubmit"
    assert automation.server_name == "spotify"
    assert automation.tool_name == "play"
    assert automation.tool_input == {"uri": "spotify:track:123"}


def test_acp_agent_config_timeout_s_defaults_to_none() -> None:
    config = AcpAgentConfig(name="claude-code", executable="/usr/local/bin/claude")

    assert config.timeout_s is None


def test_acp_agent_config_timeout_s_roundtrip_through_save_and_load(tmp_path: Path) -> None:
    settings = AppSettings(
        acp_agents=[
            AcpAgentConfig(
                name="claude-code",
                executable="/usr/local/bin/claude",
                args=["--acp"],
                timeout_s=120.0,
            )
        ]
    )

    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded == settings
    assert loaded.acp_agents[0].timeout_s == 120.0


def test_long_task_config_defaults() -> None:
    long_tasks = LongTaskConfig()

    assert long_tasks.spawn_agents_deadline_s == 600.0
    assert long_tasks.acp_delegate_default_timeout_s == 900.0


def test_app_settings_long_tasks_roundtrip_through_save_and_load(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.long_tasks = LongTaskConfig(
        spawn_agents_deadline_s=120.0, acp_delegate_default_timeout_s=300.0
    )

    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded.long_tasks == settings.long_tasks


def test_memory_tuning_config_defaults_match_original_hardcoded_constants() -> None:
    tuning = MemoryTuningConfig()

    assert tuning.retrieval_min_confidence == 0.7
    assert tuning.base_similarity_threshold == 0.80
    assert tuning.emotional_confidence_threshold == 0.75
    assert tuning.compaction_token_threshold == 60_000
    assert tuning.compaction_keep_recent == 6
    assert tuning.screen_match_threshold == 0.6


def test_app_settings_memory_tuning_roundtrip_through_save_and_load(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.memory_tuning = MemoryTuningConfig(
        retrieval_min_confidence=0.5,
        base_similarity_threshold=0.9,
        emotional_confidence_threshold=0.6,
        compaction_token_threshold=30_000,
        compaction_keep_recent=3,
        screen_match_threshold=0.8,
    )

    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded.memory_tuning == settings.memory_tuning


def test_load_settings_with_vault_migrates_legacy_plaintext_api_keys(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    legacy = AppSettings()
    legacy.model_router.anthropic = ProviderConfig(
        api_key="sk-ant-legacy", models={ModelTier.MEDIUM: "claude-sonnet-4-6"}
    )
    legacy.image_generation = ImageGenerationConfig(api_key="sk-image-legacy")
    legacy.save(settings_path)

    vault = _make_vault(tmp_path)
    try:
        loaded = load_settings_with_vault(settings_path, vault)

        assert loaded.model_router.anthropic.api_key == "sk-ant-legacy"
        assert loaded.image_generation.api_key == "sk-image-legacy"

        on_disk = json.loads(settings_path.read_text(encoding="utf-8"))
        assert on_disk["model_router"]["anthropic"]["api_key"].startswith("vault-ref:")
        assert on_disk["image_generation"]["api_key"].startswith("vault-ref:")
        assert "sk-ant-legacy" not in settings_path.read_text(encoding="utf-8")
        assert "sk-image-legacy" not in settings_path.read_text(encoding="utf-8")
    finally:
        vault.close()


def test_save_settings_with_vault_stores_plaintext_in_vault_not_on_disk(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.model_router.qwen = ProviderConfig(
        api_key="sk-qwen-plain", models={ModelTier.FAST: "qwen3-vl-plus"}
    )

    vault = _make_vault(tmp_path)
    try:
        save_settings_with_vault(settings, settings_path, vault)

        assert settings.model_router.qwen.api_key == "sk-qwen-plain"

        on_disk_text = settings_path.read_text(encoding="utf-8")
        assert "sk-qwen-plain" not in on_disk_text
        on_disk = json.loads(on_disk_text)
        vault_ref = on_disk["model_router"]["qwen"]["api_key"]
        assert vault_ref.startswith("vault-ref:")
        assert vault.get(vault_ref.removeprefix("vault-ref:")) == "sk-qwen-plain"
    finally:
        vault.close()


def test_load_settings_with_vault_roundtrips_after_save(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.model_router.openai = ProviderConfig(
        api_key="sk-openai-plain", models={ModelTier.HEAVY: "gpt-5"}
    )

    vault = _make_vault(tmp_path)
    try:
        save_settings_with_vault(settings, settings_path, vault)
        reloaded = load_settings_with_vault(settings_path, vault)

        assert reloaded.model_router.openai.api_key == "sk-openai-plain"
        on_disk = json.loads(settings_path.read_text(encoding="utf-8"))
        assert on_disk["model_router"]["openai"]["api_key"].startswith("vault-ref:")
    finally:
        vault.close()


def test_kiosk_config_defaults_to_xff() -> None:
    assert KioskConfig().default_pet == "xff"


def test_kiosk_config_defaults_to_scaled_up_and_rotated() -> None:
    config = KioskConfig()
    assert config.character_scale == 2.5
    assert config.rotate_90_clockwise is True


def test_app_settings_kiosk_roundtrip_through_save_and_load(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.kiosk.default_pet = "miku_pixel"

    path = tmp_path / "settings.json"
    settings.save(path)
    loaded = AppSettings.load(path)

    assert loaded.kiosk.default_pet == "miku_pixel"
