"""SettingsPanel 的回归测试：验证表单加载/收集往返，以及三个列表编辑器的增删改逻辑。"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QFormLayout
from qfluentwidgets import InfoBar, MessageBox

from miku_on_desk.config.settings import (
    AcpAgentConfig,
    AgentProfileConfig,
    AppSettings,
    ComputerUseConfig,
    McpServerConfig,
    McpTransport,
    ModelTier,
    PersonaConfig,
    ProactiveConfig,
    ProviderConfig,
    ProviderName,
    VoiceCloningConfig,
)
from miku_on_desk.face.ui.settings_panel import (
    _QWEN_BASE_URL,
    _QWEN_FAST_MODEL,
    SettingsPanel,
)


def _accept_message_box(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MessageBox, "exec", lambda self: 1)


def _reject_message_box(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MessageBox, "exec", lambda self: 0)


def test_provider_fields_load_from_settings(qapp: QApplication) -> None:
    settings = AppSettings()
    settings.model_router.anthropic = ProviderConfig(
        api_key="sk-ant", base_url="https://api.anthropic.com", models={ModelTier.FAST: "haiku"}
    )

    panel = SettingsPanel(settings, Path("unused.json"))
    widgets = panel._provider_widgets[ProviderName.ANTHROPIC]

    assert widgets.api_key.text() == "sk-ant"
    assert widgets.base_url.text() == "https://api.anthropic.com"
    assert widgets.model_edits[ModelTier.FAST].text() == "haiku"


def test_all_tabs_registered_as_fluent_sub_interfaces(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    assert panel.stackedWidget.count() == 13


def test_all_tabs_have_nondegenerate_size_after_switching_while_visible(
    qapp: QApplication,
) -> None:
    """回归：真正的空白 tab 根因是 pop 动画期间基于旧/默认尺寸渲染的一帧——构造后但
    未 show() 之前，qfluentwidgets 的 StackedWidget 根本不会给任何子页面布局（Qt 对
    不可见窗口的子控件跳过真实布局计算），所以这里必须先 show() 再逐个切换验证；
    关键修复是 ``setAnimationEnabled(False)``，让每次切换都走同步的
    ``QStackedWidget.setCurrentIndex``，不再有动画期间的过渡帧。
    """
    panel = SettingsPanel(AppSettings(), Path("unused.json"))
    assert panel.stackedWidget.currentIndex() == 0
    assert panel.stackedWidget.isAnimationEnabled() is False

    panel.show()
    qapp.processEvents()
    try:
        for index in range(panel.stackedWidget.count()):
            panel.stackedWidget.setCurrentIndex(index, popOut=False)
            qapp.processEvents()
            widget = panel.stackedWidget.widget(index)
            assert widget.width() > 100
            assert widget.height() > 100
    finally:
        panel.close()


def test_navigation_interface_is_not_collapsible(qapp: QApplication) -> None:
    from qfluentwidgets.components.navigation.navigation_panel import NavigationDisplayMode

    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    assert panel.navigationInterface.panel._isCollapsible is False
    assert panel.navigationInterface.panel.displayMode == NavigationDisplayMode.EXPAND


def test_current_settings_collects_edited_provider_fields(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))
    widgets = panel._provider_widgets[ProviderName.OPENAI]

    widgets.api_key.setText("sk-openai")
    widgets.base_url.setText("https://api.openai.com/v1")
    widgets.model_edits[ModelTier.HEAVY].setText("gpt-5")

    collected = panel.current_settings().model_router.openai
    assert collected.api_key == "sk-openai"
    assert collected.base_url == "https://api.openai.com/v1"
    assert collected.models == {ModelTier.HEAVY: "gpt-5"}


def test_qwen_provider_card_only_shows_api_key_field(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))
    widgets = panel._provider_widgets[ProviderName.QWEN]

    assert widgets.base_url.text() == _QWEN_BASE_URL
    assert widgets.model_edits[ModelTier.FAST].text() == _QWEN_FAST_MODEL
    assert all(
        edit.text() == ""
        for tier, edit in widgets.model_edits.items()
        if tier is not ModelTier.FAST
    )

    form = widgets.api_key.parentWidget().layout()
    assert isinstance(form, QFormLayout)
    assert form.rowCount() == 2


def test_current_settings_collects_qwen_api_key_with_fixed_defaults(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))
    widgets = panel._provider_widgets[ProviderName.QWEN]

    widgets.api_key.setText("sk-qwen")

    collected = panel.current_settings().model_router.qwen
    assert collected.api_key == "sk-qwen"
    assert collected.base_url == _QWEN_BASE_URL
    assert collected.models == {ModelTier.FAST: _QWEN_FAST_MODEL}
    assert collected.enabled is True


def test_persona_fields_load_from_settings(qapp: QApplication) -> None:
    settings = AppSettings()
    settings.persona = PersonaConfig(name="小明", role="程序员助手", personality="冷静克制")

    panel = SettingsPanel(settings, Path("unused.json"))

    assert panel._persona_name_edit.text() == "小明"
    assert panel._persona_role_edit.text() == "程序员助手"
    assert panel._persona_personality_edit.toPlainText() == "冷静克制"


def test_current_settings_collects_edited_persona(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    panel._persona_name_edit.setText("小明")
    panel._persona_role_edit.setText("程序员助手")
    panel._persona_personality_edit.setPlainText("冷静克制")

    persona = panel.current_settings().persona
    assert persona == PersonaConfig(name="小明", role="程序员助手", personality="冷静克制")


def test_current_settings_collects_permissions(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    panel._trusted_mode_box.setChecked(True)
    panel._default_decision_combo.setCurrentText("deny")
    panel._builtin_tool_combos["computer_input"].setCurrentText("总是允许")
    panel._builtin_tool_combos["screen_analyze"].setCurrentText("总是允许")
    panel._builtin_tool_combos["acp_delegate"].setCurrentText("总是禁止")
    panel._allowed_dirs_edit.setPlainText("/tmp\n/home/user")

    permissions = panel.current_settings().permissions
    assert permissions.trusted_mode is True
    assert permissions.default_decision == "deny"
    assert permissions.allowed_tools == ["computer_input", "screen_analyze"]
    assert permissions.denied_tools == ["acp_delegate"]
    assert permissions.allowed_dirs == [Path("/tmp"), Path("/home/user")]


def test_long_task_fields_load_from_settings(qapp: QApplication) -> None:
    from miku_on_desk.config.settings import LongTaskConfig

    settings = AppSettings()
    settings.long_tasks = LongTaskConfig(
        spawn_agents_deadline_s=120.0, acp_delegate_default_timeout_s=300.0
    )

    panel = SettingsPanel(settings, Path("unused.json"))

    assert panel._spawn_agents_deadline_edit.text() == "120.0"
    assert panel._acp_delegate_timeout_edit.text() == "300.0"


def test_current_settings_collects_edited_long_task_fields(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    panel._spawn_agents_deadline_edit.setText("120")
    panel._acp_delegate_timeout_edit.setText("300")

    long_tasks = panel.current_settings().long_tasks
    assert long_tasks.spawn_agents_deadline_s == 120.0
    assert long_tasks.acp_delegate_default_timeout_s == 300.0


def test_advanced_fields_load_from_settings(qapp: QApplication) -> None:
    from miku_on_desk.config.settings import LoopBehaviorConfig

    settings = AppSettings()
    settings.model_router.enable_cross_provider_fallback = True
    settings.hook_server.include_experimental = True
    settings.loop_behavior = LoopBehaviorConfig(
        max_tool_rounds=42,
        idle_timeout_s=30.0,
        hard_timeout_s=300.0,
        budget_caution_remaining=5,
        budget_critical_remaining=2,
        deadline_s=90.0,
        time_caution_remaining_s=15.0,
        time_critical_remaining_s=5.0,
    )

    panel = SettingsPanel(settings, Path("unused.json"))

    assert panel._enable_cross_provider_fallback_box.isChecked() is True
    assert panel._include_experimental_box.isChecked() is True
    assert panel._max_tool_rounds_edit.text() == "42"
    assert panel._idle_timeout_edit.text() == "30.0"
    assert panel._hard_timeout_edit.text() == "300.0"
    assert panel._budget_caution_remaining_edit.text() == "5"
    assert panel._budget_critical_remaining_edit.text() == "2"
    assert panel._deadline_edit.text() == "90.0"
    assert panel._time_caution_remaining_edit.text() == "15.0"
    assert panel._time_critical_remaining_edit.text() == "5.0"


def test_advanced_fields_load_empty_deadline_when_none(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    assert panel._deadline_edit.text() == ""


def test_current_settings_collects_edited_advanced_fields(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    panel._enable_cross_provider_fallback_box.setChecked(True)
    panel._include_experimental_box.setChecked(True)
    panel._max_tool_rounds_edit.setText("42")
    panel._idle_timeout_edit.setText("30")
    panel._hard_timeout_edit.setText("300")
    panel._budget_caution_remaining_edit.setText("5")
    panel._budget_critical_remaining_edit.setText("2")
    panel._deadline_edit.setText("90")
    panel._time_caution_remaining_edit.setText("15")
    panel._time_critical_remaining_edit.setText("5")

    settings = panel.current_settings()
    assert settings.model_router.enable_cross_provider_fallback is True
    assert settings.hook_server.include_experimental is True
    loop_behavior = settings.loop_behavior
    assert loop_behavior.max_tool_rounds == 42
    assert loop_behavior.idle_timeout_s == 30.0
    assert loop_behavior.hard_timeout_s == 300.0
    assert loop_behavior.budget_caution_remaining == 5
    assert loop_behavior.budget_critical_remaining == 2
    assert loop_behavior.deadline_s == 90.0
    assert loop_behavior.time_caution_remaining_s == 15.0
    assert loop_behavior.time_critical_remaining_s == 5.0


def test_current_settings_collects_empty_deadline_as_none(qapp: QApplication) -> None:
    from miku_on_desk.config.settings import LoopBehaviorConfig

    settings = AppSettings()
    settings.loop_behavior = LoopBehaviorConfig(deadline_s=90.0)
    panel = SettingsPanel(settings, Path("unused.json"))

    panel._deadline_edit.setText("")

    assert panel.current_settings().loop_behavior.deadline_s is None


def test_permissions_combo_default_choice_omits_tool_from_both_lists(
    qapp: QApplication,
) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    permissions = panel.current_settings().permissions
    assert permissions.allowed_tools == []
    assert permissions.denied_tools == []


def test_permissions_collect_preserves_preexisting_non_builtin_tool_names(
    qapp: QApplication,
) -> None:
    settings = AppSettings()
    settings.permissions.allowed_tools = ["mcp_weather_lookup"]
    settings.permissions.denied_tools = ["mcp_dangerous_tool"]
    panel = SettingsPanel(settings, Path("unused.json"))

    panel._builtin_tool_combos["skill"].setCurrentText("总是允许")

    permissions = panel.current_settings().permissions
    assert "mcp_weather_lookup" in permissions.allowed_tools
    assert "skill" in permissions.allowed_tools
    assert permissions.denied_tools == ["mcp_dangerous_tool"]


def test_load_permissions_reflects_denied_over_allowed_priority(qapp: QApplication) -> None:
    settings = AppSettings()
    settings.permissions.allowed_tools = ["skill"]
    settings.permissions.denied_tools = ["skill"]

    panel = SettingsPanel(settings, Path("unused.json"))

    assert panel._builtin_tool_combos["skill"].currentText() == "总是禁止"


def test_current_settings_collects_skills_dir(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    panel._skills_dir_edit.setText("/opt/skills")

    assert panel.current_settings().skills_dir == Path("/opt/skills")


def test_browse_skills_dir_button_fills_edit_with_selected_directory(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *args, **kwargs: "/opt/chosen-skills"
    )

    panel._on_browse_skills_dir()

    assert panel._skills_dir_edit.text() == "/opt/chosen-skills"


def test_browse_skills_dir_button_ignores_cancelled_dialog(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))
    panel._skills_dir_edit.setText("/opt/skills")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: "")

    panel._on_browse_skills_dir()

    assert panel._skills_dir_edit.text() == "/opt/skills"


def test_current_settings_collects_memory_dir(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    panel._memory_dir_edit.setText("/opt/memory")

    assert panel.current_settings().memory_dir == Path("/opt/memory")


def test_browse_memory_dir_button_fills_edit_with_selected_directory(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *args, **kwargs: "/opt/chosen-memory"
    )

    panel._on_browse_memory_dir()

    assert panel._memory_dir_edit.text() == "/opt/chosen-memory"


def test_browse_memory_dir_button_ignores_cancelled_dialog(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))
    panel._memory_dir_edit.setText("/opt/memory")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: "")

    panel._on_browse_memory_dir()

    assert panel._memory_dir_edit.text() == "/opt/memory"


def test_current_settings_collects_window_fields(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    panel._window_x_edit.setText("42")
    panel._window_y_edit.setText("7")
    panel._window_scale_edit.setText("1.5")
    panel._window_always_on_top_box.setChecked(False)

    window = panel.current_settings().window
    assert window.x == 42
    assert window.y == 7
    assert window.scale == 1.5
    assert window.always_on_top is False


def test_proactive_fields_load_from_settings(qapp: QApplication) -> None:
    settings = AppSettings()
    settings.proactive = ProactiveConfig(
        enabled=True,
        min_interval_s=60,
        max_interval_s=120,
        idle_threshold_s=30,
        quiet_hours_start="22:00",
        quiet_hours_end="06:00",
        max_daily_triggers=5,
    )

    panel = SettingsPanel(settings, Path("unused.json"))

    assert panel._proactive_enabled_box.isChecked() is True
    assert panel._proactive_min_interval_edit.text() == "60"
    assert panel._proactive_max_interval_edit.text() == "120"
    assert panel._proactive_idle_threshold_edit.text() == "30"
    assert panel._proactive_quiet_start_edit.text() == "22:00"
    assert panel._proactive_quiet_end_edit.text() == "06:00"
    assert panel._proactive_max_daily_edit.text() == "5"


def test_proactive_fields_load_empty_quiet_hours_when_unset(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    assert panel._proactive_quiet_start_edit.text() == ""
    assert panel._proactive_quiet_end_edit.text() == ""


def test_current_settings_collects_edited_proactive_fields(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    panel._proactive_enabled_box.setChecked(True)
    panel._proactive_min_interval_edit.setText("60")
    panel._proactive_max_interval_edit.setText("120")
    panel._proactive_idle_threshold_edit.setText("30")
    panel._proactive_quiet_start_edit.setText("22:00")
    panel._proactive_quiet_end_edit.setText("06:00")
    panel._proactive_max_daily_edit.setText("5")

    proactive = panel.current_settings().proactive
    assert proactive == ProactiveConfig(
        enabled=True,
        min_interval_s=60,
        max_interval_s=120,
        idle_threshold_s=30,
        quiet_hours_start="22:00",
        quiet_hours_end="06:00",
        max_daily_triggers=5,
    )


def test_current_settings_collects_empty_quiet_hours_as_none(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    panel._proactive_quiet_start_edit.setText("")
    panel._proactive_quiet_end_edit.setText("")

    proactive = panel.current_settings().proactive
    assert proactive.quiet_hours_start is None
    assert proactive.quiet_hours_end is None


def test_computer_use_fields_load_from_settings(qapp: QApplication) -> None:
    settings = AppSettings()
    settings.computer_use = ComputerUseConfig(enabled=True, settle_delay_s=1.5)

    panel = SettingsPanel(settings, Path("unused.json"))

    assert panel._computer_use_enabled_box.isChecked() is True
    assert panel._computer_use_settle_delay_edit.text() == "1.5"


def test_current_settings_collects_edited_computer_use_fields(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    panel._computer_use_enabled_box.setChecked(True)
    panel._computer_use_settle_delay_edit.setText("1.5")

    computer_use = panel.current_settings().computer_use
    assert computer_use == ComputerUseConfig(enabled=True, settle_delay_s=1.5)


def test_voice_cloning_fields_load_from_settings(qapp: QApplication) -> None:
    settings = AppSettings()
    settings.voice_cloning = VoiceCloningConfig(
        elevenlabs_api_key="el-key", elevenlabs_base_url="https://el.example.com"
    )

    panel = SettingsPanel(settings, Path("unused.json"))

    assert panel._voice_cloning_api_key_edit.text() == "el-key"
    assert panel._voice_cloning_base_url_edit.text() == "https://el.example.com"


def test_current_settings_collects_edited_voice_cloning_fields(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    panel._voice_cloning_api_key_edit.setText("el-key")
    panel._voice_cloning_base_url_edit.setText("https://el.example.com")

    voice_cloning = panel.current_settings().voice_cloning
    assert voice_cloning == VoiceCloningConfig(
        elevenlabs_api_key="el-key", elevenlabs_base_url="https://el.example.com"
    )


def test_current_settings_collects_empty_voice_cloning_fields_as_none(
    qapp: QApplication,
) -> None:
    settings = AppSettings()
    settings.voice_cloning = VoiceCloningConfig(
        elevenlabs_api_key="el-key", elevenlabs_base_url="https://el.example.com"
    )
    panel = SettingsPanel(settings, Path("unused.json"))

    panel._voice_cloning_api_key_edit.setText("")
    panel._voice_cloning_base_url_edit.setText("")

    voice_cloning = panel.current_settings().voice_cloning
    assert voice_cloning.elevenlabs_api_key is None
    assert voice_cloning.elevenlabs_base_url is None


def test_shortcuts_tab_defaults_to_shift_ctrl_y_and_n(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    assert panel._confirm_yes_edit.keySequence().toString() == "Ctrl+Shift+Y"
    assert panel._confirm_no_edit.keySequence().toString() == "Ctrl+Shift+N"


def test_current_settings_collects_edited_shortcuts(qapp: QApplication) -> None:
    from PySide6.QtGui import QKeySequence

    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    panel._confirm_yes_edit.setKeySequence(QKeySequence("Ctrl+Alt+Y"))
    panel._confirm_no_edit.setKeySequence(QKeySequence("Ctrl+Alt+N"))

    shortcuts = panel.current_settings().shortcuts
    assert shortcuts.confirm_yes == "Ctrl+Alt+Y"
    assert shortcuts.confirm_no == "Ctrl+Alt+N"


def test_save_button_writes_settings_to_disk_and_emits_signal(
    qapp: QApplication, tmp_path: Path
) -> None:
    settings_path = tmp_path / "settings.json"
    panel = SettingsPanel(AppSettings(), settings_path)
    panel._window_x_edit.setText("999")
    saved: list[AppSettings] = []
    panel.settings_saved.connect(saved.append)

    panel._on_save_clicked()

    assert settings_path.exists()
    assert saved[0].window.x == 999
    assert AppSettings.load(settings_path).window.x == 999


def test_save_button_shows_success_info_bar(qapp: QApplication, tmp_path: Path) -> None:
    panel = SettingsPanel(AppSettings(), tmp_path / "settings.json")

    panel._on_save_clicked()

    assert panel.findChildren(InfoBar)


def test_mcp_editor_add_edit_and_remove(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _accept_message_box(monkeypatch)
    configs: list[McpServerConfig] = []
    panel = SettingsPanel(AppSettings(mcp_servers=configs), Path("unused.json"))
    editor = panel._mcp_editor

    editor._name_edit.setText("weather")
    editor._command_edit.setText("weather-mcp")
    editor._args_edit.setText("--port, 8080")
    editor._env_edit.setPlainText("API_KEY=abc")
    editor._on_add_or_update()

    assert len(editor._configs) == 1
    added = editor._configs[0]
    assert added.name == "weather"
    assert added.command == "weather-mcp"
    assert added.args == ["--port", "8080"]
    assert added.env == {"API_KEY": "abc"}

    editor._list.setCurrentRow(0)
    editor._command_edit.setText("weather-mcp-v2")
    editor._on_add_or_update()
    assert len(editor._configs) == 1
    assert editor._configs[0].command == "weather-mcp-v2"

    editor._list.setCurrentRow(0)
    editor._on_remove()
    assert editor._configs == []


def test_mcp_editor_transport_switch_toggles_field_visibility(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(mcp_servers=[]), Path("unused.json"))
    editor = panel._mcp_editor
    form = editor._form

    assert form.isRowVisible(editor._command_edit) is True
    assert form.isRowVisible(editor._args_edit) is True
    assert form.isRowVisible(editor._env_edit) is True
    assert form.isRowVisible(editor._url_edit) is False
    assert form.isRowVisible(editor._headers_edit) is False

    editor._transport_combo.setCurrentIndex(list(McpTransport).index(McpTransport.SSE))

    assert form.isRowVisible(editor._command_edit) is False
    assert form.isRowVisible(editor._args_edit) is False
    assert form.isRowVisible(editor._env_edit) is False
    assert form.isRowVisible(editor._url_edit) is True
    assert form.isRowVisible(editor._headers_edit) is True


def test_mcp_editor_add_remote_transport_config(qapp: QApplication) -> None:
    configs: list[McpServerConfig] = []
    panel = SettingsPanel(AppSettings(mcp_servers=configs), Path("unused.json"))
    editor = panel._mcp_editor

    editor._name_edit.setText("remote-weather")
    editor._transport_combo.setCurrentIndex(
        list(McpTransport).index(McpTransport.STREAMABLE_HTTP)
    )
    editor._url_edit.setText("https://example.com/mcp")
    editor._headers_edit.setPlainText("Authorization=Bearer abc123")
    editor._on_add_or_update()

    assert len(editor._configs) == 1
    added = editor._configs[0]
    assert added.transport is McpTransport.STREAMABLE_HTTP
    assert added.url == "https://example.com/mcp"
    assert added.headers == {"Authorization": "Bearer abc123"}
    assert added.command is None


def test_selecting_remote_config_populates_transport_and_url(qapp: QApplication) -> None:
    configs = [
        McpServerConfig(
            name="remote",
            transport=McpTransport.SSE,
            url="https://example.com/sse",
            headers={"X-Token": "xyz"},
        )
    ]
    panel = SettingsPanel(AppSettings(mcp_servers=configs), Path("unused.json"))
    editor = panel._mcp_editor

    editor._list.setCurrentRow(0)

    assert editor._transport_combo.currentIndex() == list(McpTransport).index(McpTransport.SSE)
    assert editor._url_edit.text() == "https://example.com/sse"
    assert editor._headers_edit.toPlainText() == "X-Token=xyz"


def test_mcp_editor_trusted_checkbox_round_trips_through_config(qapp: QApplication) -> None:
    configs: list[McpServerConfig] = []
    panel = SettingsPanel(AppSettings(mcp_servers=configs), Path("unused.json"))
    editor = panel._mcp_editor

    editor._name_edit.setText("trusted-server")
    editor._command_edit.setText("trusted-mcp")
    editor._trusted_box.setChecked(True)
    editor._on_add_or_update()

    assert len(editor._configs) == 1
    assert editor._configs[0].trusted is True

    editor._list.setCurrentRow(0)
    assert editor._trusted_box.isChecked() is True

    editor._trusted_box.setChecked(False)
    editor._on_add_or_update()
    assert editor._configs[0].trusted is False


def test_agent_profile_editor_add_edit_and_remove(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _accept_message_box(monkeypatch)
    configs: list[AgentProfileConfig] = []
    panel = SettingsPanel(AppSettings(agent_profiles=configs), Path("unused.json"))
    editor = panel._agent_editor

    editor._name_edit.setText("researcher")
    editor._system_prompt_edit.setPlainText("你是一个研究员")
    editor._on_add_or_update()

    assert len(editor._configs) == 1
    assert editor._configs[0].system_prompt == "你是一个研究员"

    editor._list.setCurrentRow(0)
    editor._enabled_box.setChecked(False)
    editor._on_add_or_update()
    assert editor._configs[0].enabled is False

    editor._list.setCurrentRow(0)
    editor._on_remove()
    assert editor._configs == []


def test_acp_editor_add_edit_and_remove(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _accept_message_box(monkeypatch)
    configs: list[AcpAgentConfig] = []
    panel = SettingsPanel(AppSettings(acp_agents=configs), Path("unused.json"))
    editor = panel._acp_editor

    editor._name_edit.setText("claude-code")
    editor._executable_edit.setText("/usr/local/bin/claude")
    editor._args_edit.setText("--acp")
    editor._on_add_or_update()

    assert len(editor._configs) == 1
    added = editor._configs[0]
    assert added.executable == "/usr/local/bin/claude"
    assert added.args == ["--acp"]

    editor._list.setCurrentRow(0)
    editor._executable_edit.setText("/opt/claude")
    editor._on_add_or_update()
    assert editor._configs[0].executable == "/opt/claude"

    editor._list.setCurrentRow(0)
    editor._on_remove()
    assert editor._configs == []


def test_acp_editor_timeout_field_defaults_empty_and_roundtrips(qapp: QApplication) -> None:
    configs: list[AcpAgentConfig] = []
    panel = SettingsPanel(AppSettings(acp_agents=configs), Path("unused.json"))
    editor = panel._acp_editor

    editor._name_edit.setText("claude-code")
    editor._executable_edit.setText("/usr/local/bin/claude")
    editor._on_add_or_update()
    assert editor._configs[0].timeout_s is None

    editor._list.setCurrentRow(0)
    assert editor._timeout_edit.text() == ""

    editor._timeout_edit.setText("45")
    editor._on_add_or_update()
    assert editor._configs[0].timeout_s == 45.0

    editor._list.setCurrentRow(0)
    assert editor._timeout_edit.text() == "45.0"


def test_acp_editor_timeout_field_ignores_unparseable_input(qapp: QApplication) -> None:
    configs: list[AcpAgentConfig] = []
    panel = SettingsPanel(AppSettings(acp_agents=configs), Path("unused.json"))
    editor = panel._acp_editor

    editor._name_edit.setText("claude-code")
    editor._executable_edit.setText("/usr/local/bin/claude")
    editor._timeout_edit.setText("not-a-number")
    editor._on_add_or_update()

    assert editor._configs[0].timeout_s is None


def test_selecting_list_item_populates_form(qapp: QApplication) -> None:
    configs = [McpServerConfig(name="a", command="cmd-a", args=["x"], enabled=False)]
    panel = SettingsPanel(AppSettings(mcp_servers=configs), Path("unused.json"))
    editor = panel._mcp_editor

    editor._list.setCurrentRow(0)

    assert editor._name_edit.text() == "a"
    assert editor._command_edit.text() == "cmd-a"
    assert editor._args_edit.text() == "x"
    assert editor._enabled_box.isChecked() is False


def test_skills_tab_form_fields_grow_to_fill_width(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    forms = panel.findChildren(QFormLayout)
    assert len(forms) > 0
    for form in forms:
        assert form.fieldGrowthPolicy() == QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow


def test_list_editors_have_bordered_fixed_height_list_widget(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    for editor in (panel._mcp_editor, panel._agent_editor, panel._acp_editor):
        assert editor._list.height() == 120
        assert editor._list.property("lightCustomQss")
        assert editor._list.property("darkCustomQss")


def test_invalid_numeric_input_shows_warning_and_clears_on_valid_input(
    qapp: QApplication,
) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))
    assert panel._window_x_warning.isHidden()
    assert not panel._window_x_edit.property("lightCustomQss")

    panel._window_x_edit.setText("not-a-number")

    assert not panel._window_x_warning.isHidden()
    assert "将使用默认值" in panel._window_x_warning.text()
    assert panel._window_x_edit.property("lightCustomQss")

    panel._window_x_edit.setText("42")

    assert panel._window_x_warning.isHidden()
    assert not panel._window_x_edit.property("lightCustomQss")


def test_invalid_float_numeric_input_shows_warning(qapp: QApplication) -> None:
    panel = SettingsPanel(AppSettings(), Path("unused.json"))

    panel._window_scale_edit.setText("not-a-float")

    assert not panel._window_scale_warning.isHidden()

    panel._window_scale_edit.setText("1.5")

    assert panel._window_scale_warning.isHidden()


def test_mcp_editor_remove_does_nothing_when_confirmation_declined(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reject_message_box(monkeypatch)
    configs = [McpServerConfig(name="weather", command="weather-mcp")]
    panel = SettingsPanel(AppSettings(mcp_servers=configs), Path("unused.json"))
    editor = panel._mcp_editor

    editor._list.setCurrentRow(0)
    editor._on_remove()

    assert len(editor._configs) == 1


def test_agent_profile_editor_remove_does_nothing_when_confirmation_declined(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reject_message_box(monkeypatch)
    configs = [AgentProfileConfig(name="researcher", system_prompt="p")]
    panel = SettingsPanel(AppSettings(agent_profiles=configs), Path("unused.json"))
    editor = panel._agent_editor

    editor._list.setCurrentRow(0)
    editor._on_remove()

    assert len(editor._configs) == 1


def test_acp_editor_remove_does_nothing_when_confirmation_declined(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reject_message_box(monkeypatch)
    configs = [AcpAgentConfig(name="claude-code", executable="/usr/local/bin/claude")]
    panel = SettingsPanel(AppSettings(acp_agents=configs), Path("unused.json"))
    editor = panel._acp_editor

    editor._list.setCurrentRow(0)
    editor._on_remove()

    assert len(editor._configs) == 1
