"""设置面板：编辑 Provider 凭证/模型、权限、MCP/Agent/ACP 列表与窗口配置，点击保存后
落盘到 ``settings.json``。

三个列表型配置（MCP server / Agent profile / ACP agent）各自的字段形状不同（有的带 env，
有的带 system_prompt），没有抽出通用的列表编辑基类——三份具体实现虽然结构相似，但共享的
只是"列表+表单+增删"这几行 Qt 接线，硬拆出一个基类反而要为字段差异引入泛型/回调间接层，
不划算。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QKeySequenceEdit,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CaptionLabel,
    CheckBox,
    ComboBox,
    FluentIcon,
    FluentWindow,
    HeaderCardWidget,
    InfoBar,
    LineEdit,
    ListWidget,
    MessageBox,
    PlainTextEdit,
    PrimaryPushButton,
    PushButton,
    SingleDirectionScrollArea,
    setCustomStyleSheet,
)

from miku_on_desk.config.settings import (
    AcpAgentConfig,
    AgentProfileConfig,
    AppSettings,
    McpServerConfig,
    McpTransport,
    ModelTier,
    PersonaConfig,
    ProviderConfig,
    ProviderName,
    save_settings_with_vault,
)
from miku_on_desk.face.ui.theme import RADIUS_MD, WARNING_COLOR

if TYPE_CHECKING:
    from miku_on_desk.brain.secrets.vault import SecretVault


_PROVIDER_LABELS = {
    ProviderName.ANTHROPIC: "Anthropic",
    ProviderName.OPENAI: "OpenAI 兼容",
    ProviderName.GEMINI: "Gemini",
    ProviderName.QWEN: "Qwen（DashScope）",
}

_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_QWEN_FAST_MODEL = "qwen3-vl-plus"

_MCP_TRANSPORT_LABELS: dict[McpTransport, str] = {
    McpTransport.STDIO: "本机命令（stdio）",
    McpTransport.SSE: "远程 SSE",
    McpTransport.STREAMABLE_HTTP: "远程 Streamable HTTP",
}

_BUILTIN_TOOLS: list[tuple[str, str]] = [
    ("computer_input", "操作电脑（点击/输入/开应用）"),
    ("screen_analyze", "查看屏幕内容"),
    ("skill", "执行预先写好的 Skill"),
    ("spawn_agents", "派发任务给内部子 Agent"),
    ("acp_delegate", "外包给本机其他编码 Agent"),
]

_TOOL_CHOICE_DEFAULT = "跟随默认"
_TOOL_CHOICE_ALLOW = "总是允许"
_TOOL_CHOICE_DENY = "总是禁止"
_TOOL_CHOICES = [_TOOL_CHOICE_DEFAULT, _TOOL_CHOICE_ALLOW, _TOOL_CHOICE_DENY]

_LIST_BORDER_LIGHT_QSS = (
    f"ListWidget{{border: 1px solid rgba(0, 0, 0, 45); border-radius: {RADIUS_MD}px;}}"
)
_LIST_BORDER_DARK_QSS = (
    f"ListWidget{{border: 1px solid rgba(255, 255, 255, 45); border-radius: {RADIUS_MD}px;}}"
)
_NUMERIC_WARNING_QSS = (
    f"LineEdit{{border: 1px solid {WARNING_COLOR}; border-radius: {RADIUS_MD}px;}}"
)


def _style_list_widget(list_widget: ListWidget) -> None:
    """给空列表加一圈可见描边，否则 0 项时是一块没有任何边界的空白区域。"""
    setCustomStyleSheet(list_widget, _LIST_BORDER_LIGHT_QSS, _LIST_BORDER_DARK_QSS)
    list_widget.setFixedHeight(120)


def _lines_to_list(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _list_to_lines(items: list[str]) -> str:
    return "\n".join(items)


def _csv_to_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _list_to_csv(items: list[str]) -> str:
    return ", ".join(items)


def _env_lines_to_dict(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in _lines_to_list(text):
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _env_dict_to_lines(env: dict[str, str]) -> str:
    return "\n".join(f"{key}={value}" for key, value in env.items())


@dataclass
class _ProviderFormWidgets:
    api_key: LineEdit
    base_url: LineEdit
    model_edits: dict[ModelTier, LineEdit]


class SettingsPanel(FluentWindow):  # type: ignore[misc]
    """构造时深拷贝传入的 ``AppSettings``，编辑期间不影响调用方持有的原对象；点击保存
    按钮才会落盘并通过 ``settings_saved`` 通知外部（例如 main.py 据此重建 model_router）。
    """

    settings_saved = Signal(object)

    def __init__(
        self,
        settings: AppSettings,
        settings_path: Path,
        parent: QWidget | None = None,
        *,
        vault: SecretVault | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings.model_copy(deep=True)
        self._settings_path = settings_path
        self._vault = vault

        self._provider_widgets: dict[ProviderName, _ProviderFormWidgets] = {}
        self._persona_name_edit = LineEdit(self)
        self._persona_role_edit = LineEdit(self)
        self._persona_personality_edit = PlainTextEdit(self)
        self._trusted_mode_box = CheckBox("信任模式（跳过询问，直接放行）", self)
        self._default_decision_combo = ComboBox(self)
        self._builtin_tool_combos: dict[str, ComboBox] = {}
        self._allowed_dirs_edit = PlainTextEdit(self)
        self._spawn_agents_deadline_edit = LineEdit(self)
        self._acp_delegate_timeout_edit = LineEdit(self)
        self._skills_dir_edit = LineEdit(self)
        self._memory_dir_edit = LineEdit(self)
        self._window_x_edit = LineEdit(self)
        self._window_y_edit = LineEdit(self)
        self._window_scale_edit = LineEdit(self)
        self._window_always_on_top_box = CheckBox("始终置顶", self)
        self._confirm_yes_edit = QKeySequenceEdit(self)
        self._confirm_no_edit = QKeySequenceEdit(self)
        self._proactive_enabled_box = CheckBox("启用主动交互", self)
        self._proactive_min_interval_edit = LineEdit(self)
        self._proactive_max_interval_edit = LineEdit(self)
        self._proactive_idle_threshold_edit = LineEdit(self)
        self._proactive_quiet_start_edit = LineEdit(self)
        self._proactive_quiet_end_edit = LineEdit(self)
        self._proactive_max_daily_edit = LineEdit(self)
        self._enable_cross_provider_fallback_box = CheckBox("允许跨 Provider 降级", self)
        self._include_experimental_box = CheckBox("启用实验性 Hook 事件", self)
        self._max_tool_rounds_edit = LineEdit(self)
        self._idle_timeout_edit = LineEdit(self)
        self._hard_timeout_edit = LineEdit(self)
        self._budget_caution_remaining_edit = LineEdit(self)
        self._budget_critical_remaining_edit = LineEdit(self)
        self._deadline_edit = LineEdit(self)
        self._time_caution_remaining_edit = LineEdit(self)
        self._time_critical_remaining_edit = LineEdit(self)

        self._mcp_editor = _McpServerListEditor(self._settings.mcp_servers, self)
        self._agent_editor = _AgentProfileListEditor(self._settings.agent_profiles, self)
        self._acp_editor = _AcpAgentListEditor(self._settings.acp_agents, self)

        providers_tab = self._build_providers_tab()
        persona_tab = self._build_persona_tab()
        permissions_tab = self._build_permissions_tab()
        skills_tab = self._build_skills_tab()
        memory_tab = self._build_memory_tab()
        window_tab = self._build_window_tab()
        shortcuts_tab = self._build_shortcuts_tab()
        proactive_tab = self._build_proactive_tab()
        advanced_tab = self._build_advanced_tab()
        for widget, object_name in (
            (providers_tab, "providerTab"),
            (persona_tab, "personaTab"),
            (permissions_tab, "permissionsTab"),
            (self._mcp_editor, "mcpTab"),
            (skills_tab, "skillsTab"),
            (memory_tab, "memoryTab"),
            (self._agent_editor, "agentTab"),
            (self._acp_editor, "acpTab"),
            (window_tab, "windowTab"),
            (shortcuts_tab, "shortcutsTab"),
            (proactive_tab, "proactiveTab"),
            (advanced_tab, "advancedTab"),
        ):
            widget.setObjectName(object_name)
        self.addSubInterface(providers_tab, FluentIcon.CLOUD, "Provider")
        self.addSubInterface(persona_tab, FluentIcon.ROBOT, "人格")
        self.addSubInterface(permissions_tab, FluentIcon.CERTIFICATE, "权限")
        self.addSubInterface(self._mcp_editor, FluentIcon.CONNECT, "MCP")
        self.addSubInterface(skills_tab, FluentIcon.FOLDER, "Skills")
        self.addSubInterface(memory_tab, FluentIcon.HISTORY, "记忆")
        self.addSubInterface(self._agent_editor, FluentIcon.PEOPLE, "Agent")
        self.addSubInterface(self._acp_editor, FluentIcon.LINK, "ACP")
        self.addSubInterface(window_tab, FluentIcon.LAYOUT, "窗口")
        self.addSubInterface(shortcuts_tab, FluentIcon.COMMAND_PROMPT, "快捷键")
        self.addSubInterface(proactive_tab, FluentIcon.MEGAPHONE, "主动交互")
        self.addSubInterface(advanced_tab, FluentIcon.DEVELOPER_TOOLS, "高级")

        self._load_persona()
        self._load_permissions()
        self._load_skills_dir()
        self._load_memory_dir()
        self._load_window()
        self._load_shortcuts()
        self._load_proactive()
        self._load_advanced()

        save_button = PrimaryPushButton("保存", self)
        save_button.clicked.connect(self._on_save_clicked)

        # FluentWindow 默认布局是 navigationInterface | stackedWidget 左右两栏，没有
        # 现成的"跨页面常驻底栏"位置——把 stackedWidget 挪进一个竖直布局，和保存按钮
        # 一起塞回 widgetLayout，让保存按钮在所有 tab 下方常驻。
        self.widgetLayout.removeWidget(self.stackedWidget)
        content_layout = QVBoxLayout()
        content_layout.addWidget(self.stackedWidget, 1)
        content_layout.addWidget(save_button)
        self.widgetLayout.addLayout(content_layout)

        # 侧边导航栏默认要求窗口宽度 ≥1008px 才会展开显示文字标签，否则永远是纯图标——
        # 我们的窗口远小于这个阈值，因此强制常驻展开（并收窄展开宽度，标签都是 2-4 个
        # 汉字的短文本，不需要默认的 322px）。
        self.navigationInterface.setExpandWidth(140)
        self.navigationInterface.setCollapsible(False)
        self.resize(680, 560)

        # qfluentwidgets 的 StackedWidget 只给"当前显示中"的子页面分配真实几何尺寸，
        # 其余通过 addSubInterface 加入但从未切换成当前页的子页面停留在默认的很小尺寸，
        # 第一次切过去时基于这个尺寸渲染就会显得空白/裁切——这里在窗口达到最终尺寸后
        # 把每个子页面都当一次当前页，让它们各自完成一次真实 layout。
        self.stackedWidget.setAnimationEnabled(False)
        for index in range(self.stackedWidget.count()):
            self.stackedWidget.setCurrentIndex(index, popOut=False)
        self.stackedWidget.setCurrentIndex(0, popOut=False)

    def current_settings(self) -> AppSettings:
        """把各 tab 的编辑状态收集回一份新的 ``AppSettings``；MCP/Agent/ACP 列表在增删改
        时已经直接写回 ``self._settings``，这里只需收集表单类字段。
        """
        self._collect_providers()
        self._collect_persona()
        self._collect_permissions()
        self._collect_skills_dir()
        self._collect_memory_dir()
        self._collect_window()
        self._collect_shortcuts()
        self._collect_proactive()
        self._collect_advanced()
        return self._settings.model_copy(deep=True)

    def _add_numeric_validation(
        self,
        edit: LineEdit,
        is_valid: Callable[[str], bool],
        default: Callable[[], int | float],
    ) -> CaptionLabel:
        """给数字输入框接实时校验：非法输入时描边变警告色 + 提示将回退到的默认值，
        避免用户输入被 ``_parse_int``/``_parse_float`` 静默丢弃却毫无感知。
        """
        warning = CaptionLabel("", self)
        warning.setStyleSheet(f"color: {WARNING_COLOR};")
        warning.hide()

        def _on_text_changed(text: str) -> None:
            if is_valid(text):
                warning.hide()
                setCustomStyleSheet(edit, "", "")
            else:
                warning.setText(f"将使用默认值 {default()}")
                warning.show()
                setCustomStyleSheet(edit, _NUMERIC_WARNING_QSS, _NUMERIC_WARNING_QSS)

        edit.textChanged.connect(_on_text_changed)
        return warning

    def _build_providers_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        for name in ProviderName:
            config = self._settings.model_router.provider(name)
            layout.addWidget(self._build_provider_group(name, config))
        layout.addStretch(1)

        # 3 张 Provider 卡片堆叠起来自然高度超过 1000px，不加滚动区域的话这个 tab 的
        # minimumSizeHint 会撑大整个窗口，盖过下面 resize(680, 560) 的效果。
        scroll_area = SingleDirectionScrollArea(self)
        scroll_area.setWidget(container)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea{background: transparent; border: none}")
        container.setStyleSheet("QWidget{background: transparent}")
        return cast(QWidget, scroll_area)

    def _build_provider_group(self, name: ProviderName, config: ProviderConfig) -> HeaderCardWidget:
        card = HeaderCardWidget(_PROVIDER_LABELS[name], self)
        container = QWidget(card)
        form = QFormLayout(container)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        api_key_edit = LineEdit(container)
        api_key_edit.setText(config.api_key or "")
        api_key_edit.setEchoMode(LineEdit.EchoMode.Password)
        form.addRow("API Key", api_key_edit)

        base_url_edit = LineEdit(container)
        model_edits: dict[ModelTier, LineEdit] = {}

        if name is ProviderName.QWEN:
            base_url_edit.setText(config.base_url or _QWEN_BASE_URL)
            # Base URL 与各 tier 的模型名对 Qwen 固定死、不接受编辑，因此不 addRow 到表单里；
            # 但这两个 LineEdit 仍以 container 为 parent，不显式 hide() 的话 Qt 会把它们画在
            # (0, 0) 默认位置，跟其它可见字段重叠——_collect_providers() 保存时仍要读它们的值，
            # 所以只能隐藏而不能不创建。
            base_url_edit.hide()
            for tier in ModelTier:
                edit = LineEdit(container)
                default = _QWEN_FAST_MODEL if tier is ModelTier.FAST else ""
                edit.setText(config.models.get(tier) or default)
                edit.hide()
                model_edits[tier] = edit
            form.addRow(
                CaptionLabel(
                    "模型与接入点已固定为 Qwen3-VL-Plus / DashScope 国内站，无需填写", container
                )
            )
        else:
            base_url_edit.setText(config.base_url or "")
            form.addRow("Base URL", base_url_edit)
            for tier in ModelTier:
                edit = LineEdit(container)
                edit.setText(config.models.get(tier, ""))
                form.addRow(f"模型（{tier.value}）", edit)
                model_edits[tier] = edit

        self._provider_widgets[name] = _ProviderFormWidgets(
            api_key_edit, base_url_edit, model_edits
        )
        card.viewLayout.addWidget(container)
        return card

    def _collect_providers(self) -> None:
        for name, widgets in self._provider_widgets.items():
            models = {
                tier: text
                for tier, edit in widgets.model_edits.items()
                if (text := edit.text().strip())
            }
            config = ProviderConfig(
                api_key=widgets.api_key.text().strip() or None,
                base_url=widgets.base_url.text().strip() or None,
                models=models,
            )
            setattr(self._settings.model_router, name.value, config)

    def _build_persona_tab(self) -> QWidget:
        container = QWidget(self)
        form = QFormLayout(container)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("名字", self._persona_name_edit)
        form.addRow("角色", self._persona_role_edit)
        form.addRow("性格/说话风格", self._persona_personality_edit)
        caption = CaptionLabel("工具列表与确认授权规则不可在此修改，由代码固定保证安全", container)
        form.addRow(caption)
        return container

    def _load_persona(self) -> None:
        persona = self._settings.persona
        self._persona_name_edit.setText(persona.name)
        self._persona_role_edit.setText(persona.role)
        self._persona_personality_edit.setPlainText(persona.personality)

    def _collect_persona(self) -> PersonaConfig:
        persona = PersonaConfig(
            name=self._persona_name_edit.text().strip(),
            role=self._persona_role_edit.text().strip(),
            personality=self._persona_personality_edit.toPlainText().strip(),
        )
        self._settings.persona = persona
        return persona


    def _build_permissions_tab(self) -> QWidget:
        container = QWidget(self)
        form = QFormLayout(container)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow(self._trusted_mode_box)
        self._default_decision_combo.addItems(["ask", "deny"])
        form.addRow("默认决策", self._default_decision_combo)
        for name, description in _BUILTIN_TOOLS:
            combo = ComboBox(container)
            combo.addItems(_TOOL_CHOICES)
            self._builtin_tool_combos[name] = combo
            form.addRow(description, combo)
        form.addRow("允许访问的目录（每行一个）", self._allowed_dirs_edit)
        form.addRow(CaptionLabel("长时间任务超时", container))
        form.addRow("spawn_agents 整体超时（秒）", self._spawn_agents_deadline_edit)
        self._spawn_agents_deadline_warning = self._add_numeric_validation(
            self._spawn_agents_deadline_edit,
            _is_valid_float,
            lambda: self._settings.long_tasks.spawn_agents_deadline_s,
        )
        form.addRow(self._spawn_agents_deadline_warning)
        form.addRow("acp_delegate 默认超时（秒）", self._acp_delegate_timeout_edit)
        self._acp_delegate_timeout_warning = self._add_numeric_validation(
            self._acp_delegate_timeout_edit,
            _is_valid_float,
            lambda: self._settings.long_tasks.acp_delegate_default_timeout_s,
        )
        form.addRow(self._acp_delegate_timeout_warning)
        return container

    def _load_permissions(self) -> None:
        permissions = self._settings.permissions
        self._trusted_mode_box.setChecked(permissions.trusted_mode)
        self._default_decision_combo.setCurrentText(permissions.default_decision)
        for name, combo in self._builtin_tool_combos.items():
            if name in permissions.denied_tools:
                combo.setCurrentText(_TOOL_CHOICE_DENY)
            elif name in permissions.allowed_tools:
                combo.setCurrentText(_TOOL_CHOICE_ALLOW)
            else:
                combo.setCurrentText(_TOOL_CHOICE_DEFAULT)
        self._allowed_dirs_edit.setPlainText(
            _list_to_lines([str(path) for path in permissions.allowed_dirs])
        )
        long_tasks = self._settings.long_tasks
        self._spawn_agents_deadline_edit.setText(str(long_tasks.spawn_agents_deadline_s))
        self._acp_delegate_timeout_edit.setText(str(long_tasks.acp_delegate_default_timeout_s))

    def _collect_permissions(self) -> None:
        permissions = self._settings.permissions
        permissions.trusted_mode = self._trusted_mode_box.isChecked()
        decision = self._default_decision_combo.currentText()
        permissions.default_decision = "deny" if decision == "deny" else "ask"

        builtin_names = {name for name, _ in _BUILTIN_TOOLS}
        allowed = [tool for tool in permissions.allowed_tools if tool not in builtin_names]
        denied = [tool for tool in permissions.denied_tools if tool not in builtin_names]
        for name, combo in self._builtin_tool_combos.items():
            choice = combo.currentText()
            if choice == _TOOL_CHOICE_ALLOW:
                allowed.append(name)
            elif choice == _TOOL_CHOICE_DENY:
                denied.append(name)
        permissions.allowed_tools = allowed
        permissions.denied_tools = denied

        permissions.allowed_dirs = [
            Path(line) for line in _lines_to_list(self._allowed_dirs_edit.toPlainText())
        ]

        long_tasks = self._settings.long_tasks
        long_tasks.spawn_agents_deadline_s = _parse_float(
            self._spawn_agents_deadline_edit.text(), default=long_tasks.spawn_agents_deadline_s
        )
        long_tasks.acp_delegate_default_timeout_s = _parse_float(
            self._acp_delegate_timeout_edit.text(),
            default=long_tasks.acp_delegate_default_timeout_s,
        )

    def _build_skills_tab(self) -> QWidget:
        container = QWidget(self)
        form = QFormLayout(container)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        row = QHBoxLayout()
        row.addWidget(self._skills_dir_edit)
        browse_button = PushButton("浏览…", container)
        browse_button.clicked.connect(self._on_browse_skills_dir)
        row.addWidget(browse_button)
        form.addRow("Skills 目录", row)
        return container

    def _on_browse_skills_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "选择 Skills 根目录", self._skills_dir_edit.text()
        )
        if directory:
            self._skills_dir_edit.setText(directory)

    def _load_skills_dir(self) -> None:
        self._skills_dir_edit.setText(str(self._settings.skills_dir or ""))

    def _collect_skills_dir(self) -> None:
        text = self._skills_dir_edit.text().strip()
        self._settings.skills_dir = Path(text) if text else None

    def _build_memory_tab(self) -> QWidget:
        container = QWidget(self)
        form = QFormLayout(container)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        row = QHBoxLayout()
        row.addWidget(self._memory_dir_edit)
        browse_button = PushButton("浏览…", container)
        browse_button.clicked.connect(self._on_browse_memory_dir)
        row.addWidget(browse_button)
        form.addRow("记忆存储目录", row)
        return container

    def _on_browse_memory_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "选择记忆存储目录", self._memory_dir_edit.text()
        )
        if directory:
            self._memory_dir_edit.setText(directory)

    def _load_memory_dir(self) -> None:
        self._memory_dir_edit.setText(str(self._settings.memory_dir or ""))

    def _collect_memory_dir(self) -> None:
        text = self._memory_dir_edit.text().strip()
        self._settings.memory_dir = Path(text) if text else None

    def _build_window_tab(self) -> QWidget:
        container = QWidget(self)
        form = QFormLayout(container)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("X", self._window_x_edit)
        self._window_x_warning = self._add_numeric_validation(
            self._window_x_edit, _is_valid_int, lambda: self._settings.window.x
        )
        form.addRow(self._window_x_warning)
        form.addRow("Y", self._window_y_edit)
        self._window_y_warning = self._add_numeric_validation(
            self._window_y_edit, _is_valid_int, lambda: self._settings.window.y
        )
        form.addRow(self._window_y_warning)
        form.addRow("缩放", self._window_scale_edit)
        self._window_scale_warning = self._add_numeric_validation(
            self._window_scale_edit, _is_valid_float, lambda: self._settings.window.scale
        )
        form.addRow(self._window_scale_warning)
        form.addRow(self._window_always_on_top_box)
        return container

    def _load_window(self) -> None:
        window = self._settings.window
        self._window_x_edit.setText(str(window.x))
        self._window_y_edit.setText(str(window.y))
        self._window_scale_edit.setText(str(window.scale))
        self._window_always_on_top_box.setChecked(window.always_on_top)

    def _collect_window(self) -> None:
        window = self._settings.window
        window.x = _parse_int(self._window_x_edit.text(), default=window.x)
        window.y = _parse_int(self._window_y_edit.text(), default=window.y)
        window.scale = _parse_float(self._window_scale_edit.text(), default=window.scale)
        window.always_on_top = self._window_always_on_top_box.isChecked()

    def _build_shortcuts_tab(self) -> QWidget:
        container = QWidget(self)
        form = QFormLayout(container)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("确认「是」", self._confirm_yes_edit)
        form.addRow("确认「否」", self._confirm_no_edit)
        form.addRow(CaptionLabel("快捷键改动需要重启 Miku 才能生效", container))
        return container

    def _load_shortcuts(self) -> None:
        shortcuts = self._settings.shortcuts
        self._confirm_yes_edit.setKeySequence(QKeySequence(shortcuts.confirm_yes))
        self._confirm_no_edit.setKeySequence(QKeySequence(shortcuts.confirm_no))

    def _collect_shortcuts(self) -> None:
        shortcuts = self._settings.shortcuts
        shortcuts.confirm_yes = self._confirm_yes_edit.keySequence().toString()
        shortcuts.confirm_no = self._confirm_no_edit.keySequence().toString()

    def _build_proactive_tab(self) -> QWidget:
        container = QWidget(self)
        form = QFormLayout(container)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow(self._proactive_enabled_box)
        form.addRow("最小间隔（秒）", self._proactive_min_interval_edit)
        self._proactive_min_interval_warning = self._add_numeric_validation(
            self._proactive_min_interval_edit,
            _is_valid_int,
            lambda: self._settings.proactive.min_interval_s,
        )
        form.addRow(self._proactive_min_interval_warning)
        form.addRow("最大间隔（秒）", self._proactive_max_interval_edit)
        self._proactive_max_interval_warning = self._add_numeric_validation(
            self._proactive_max_interval_edit,
            _is_valid_int,
            lambda: self._settings.proactive.max_interval_s,
        )
        form.addRow(self._proactive_max_interval_warning)
        form.addRow("空闲阈值（秒）", self._proactive_idle_threshold_edit)
        self._proactive_idle_threshold_warning = self._add_numeric_validation(
            self._proactive_idle_threshold_edit,
            _is_valid_int,
            lambda: self._settings.proactive.idle_threshold_s,
        )
        form.addRow(self._proactive_idle_threshold_warning)
        form.addRow("免打扰开始（HH:MM，留空不启用）", self._proactive_quiet_start_edit)
        form.addRow("免打扰结束（HH:MM，留空不启用）", self._proactive_quiet_end_edit)
        form.addRow("每日最多触发次数", self._proactive_max_daily_edit)
        self._proactive_max_daily_warning = self._add_numeric_validation(
            self._proactive_max_daily_edit,
            _is_valid_int,
            lambda: self._settings.proactive.max_daily_triggers,
        )
        form.addRow(self._proactive_max_daily_warning)
        form.addRow(CaptionLabel("改动需要重启 Miku 才能生效", container))
        return container

    def _load_proactive(self) -> None:
        proactive = self._settings.proactive
        self._proactive_enabled_box.setChecked(proactive.enabled)
        self._proactive_min_interval_edit.setText(str(proactive.min_interval_s))
        self._proactive_max_interval_edit.setText(str(proactive.max_interval_s))
        self._proactive_idle_threshold_edit.setText(str(proactive.idle_threshold_s))
        self._proactive_quiet_start_edit.setText(proactive.quiet_hours_start or "")
        self._proactive_quiet_end_edit.setText(proactive.quiet_hours_end or "")
        self._proactive_max_daily_edit.setText(str(proactive.max_daily_triggers))

    def _collect_proactive(self) -> None:
        proactive = self._settings.proactive
        proactive.enabled = self._proactive_enabled_box.isChecked()
        proactive.min_interval_s = _parse_int(
            self._proactive_min_interval_edit.text(), default=proactive.min_interval_s
        )
        proactive.max_interval_s = _parse_int(
            self._proactive_max_interval_edit.text(), default=proactive.max_interval_s
        )
        proactive.idle_threshold_s = _parse_int(
            self._proactive_idle_threshold_edit.text(), default=proactive.idle_threshold_s
        )
        proactive.quiet_hours_start = self._proactive_quiet_start_edit.text().strip() or None
        proactive.quiet_hours_end = self._proactive_quiet_end_edit.text().strip() or None
        proactive.max_daily_triggers = _parse_int(
            self._proactive_max_daily_edit.text(), default=proactive.max_daily_triggers
        )

    def _build_advanced_tab(self) -> QWidget:
        outer = QWidget()
        layout = QVBoxLayout(outer)

        router_card = HeaderCardWidget("模型路由", self)
        router_container = QWidget(router_card)
        router_form = QFormLayout(router_container)
        router_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        router_form.addRow(self._enable_cross_provider_fallback_box)
        router_card.viewLayout.addWidget(router_container)
        layout.addWidget(router_card)

        hook_card = HeaderCardWidget("Hook Server", self)
        hook_container = QWidget(hook_card)
        hook_form = QFormLayout(hook_container)
        hook_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        hook_form.addRow(self._include_experimental_box)
        hook_card.viewLayout.addWidget(hook_container)
        layout.addWidget(hook_card)

        loop_card = HeaderCardWidget("AI 循环参数", self)
        loop_container = QWidget(loop_card)
        loop_form = QFormLayout(loop_container)
        loop_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        loop_form.addRow("最大工具调用回合数", self._max_tool_rounds_edit)
        self._max_tool_rounds_warning = self._add_numeric_validation(
            self._max_tool_rounds_edit,
            _is_valid_int,
            lambda: self._settings.loop_behavior.max_tool_rounds,
        )
        loop_form.addRow(self._max_tool_rounds_warning)

        loop_form.addRow("单轮响应空闲超时（秒）", self._idle_timeout_edit)
        self._idle_timeout_warning = self._add_numeric_validation(
            self._idle_timeout_edit,
            _is_valid_float,
            lambda: self._settings.loop_behavior.idle_timeout_s,
        )
        loop_form.addRow(self._idle_timeout_warning)

        loop_form.addRow("单轮请求硬超时（秒）", self._hard_timeout_edit)
        self._hard_timeout_warning = self._add_numeric_validation(
            self._hard_timeout_edit,
            _is_valid_float,
            lambda: self._settings.loop_behavior.hard_timeout_s,
        )
        loop_form.addRow(self._hard_timeout_warning)

        loop_form.addRow("回合预算提醒阈值（剩余回合数）", self._budget_caution_remaining_edit)
        self._budget_caution_remaining_warning = self._add_numeric_validation(
            self._budget_caution_remaining_edit,
            _is_valid_int,
            lambda: self._settings.loop_behavior.budget_caution_remaining,
        )
        loop_form.addRow(self._budget_caution_remaining_warning)

        loop_form.addRow("回合预算紧急阈值（剩余回合数）", self._budget_critical_remaining_edit)
        self._budget_critical_remaining_warning = self._add_numeric_validation(
            self._budget_critical_remaining_edit,
            _is_valid_int,
            lambda: self._settings.loop_behavior.budget_critical_remaining,
        )
        loop_form.addRow(self._budget_critical_remaining_warning)

        loop_form.addRow("墙钟截止时间（秒，留空=不限时）", self._deadline_edit)

        loop_form.addRow("时间预算提醒阈值（剩余秒数）", self._time_caution_remaining_edit)
        self._time_caution_remaining_warning = self._add_numeric_validation(
            self._time_caution_remaining_edit,
            _is_valid_float,
            lambda: self._settings.loop_behavior.time_caution_remaining_s,
        )
        loop_form.addRow(self._time_caution_remaining_warning)

        loop_form.addRow("时间预算紧急阈值（剩余秒数）", self._time_critical_remaining_edit)
        self._time_critical_remaining_warning = self._add_numeric_validation(
            self._time_critical_remaining_edit,
            _is_valid_float,
            lambda: self._settings.loop_behavior.time_critical_remaining_s,
        )
        loop_form.addRow(self._time_critical_remaining_warning)

        loop_form.addRow(CaptionLabel("改动需要重启 Miku 才能生效", loop_container))
        loop_card.viewLayout.addWidget(loop_container)
        layout.addWidget(loop_card)
        layout.addStretch(1)

        scroll_area = SingleDirectionScrollArea(self)
        scroll_area.setWidget(outer)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea{background: transparent; border: none}")
        outer.setStyleSheet("QWidget{background: transparent}")
        return cast(QWidget, scroll_area)

    def _load_advanced(self) -> None:
        self._enable_cross_provider_fallback_box.setChecked(
            self._settings.model_router.enable_cross_provider_fallback
        )
        self._include_experimental_box.setChecked(self._settings.hook_server.include_experimental)
        loop_behavior = self._settings.loop_behavior
        self._max_tool_rounds_edit.setText(str(loop_behavior.max_tool_rounds))
        self._idle_timeout_edit.setText(str(loop_behavior.idle_timeout_s))
        self._hard_timeout_edit.setText(str(loop_behavior.hard_timeout_s))
        self._budget_caution_remaining_edit.setText(str(loop_behavior.budget_caution_remaining))
        self._budget_critical_remaining_edit.setText(str(loop_behavior.budget_critical_remaining))
        self._deadline_edit.setText(
            "" if loop_behavior.deadline_s is None else str(loop_behavior.deadline_s)
        )
        self._time_caution_remaining_edit.setText(str(loop_behavior.time_caution_remaining_s))
        self._time_critical_remaining_edit.setText(str(loop_behavior.time_critical_remaining_s))

    def _collect_advanced(self) -> None:
        self._settings.model_router.enable_cross_provider_fallback = (
            self._enable_cross_provider_fallback_box.isChecked()
        )
        self._settings.hook_server.include_experimental = (
            self._include_experimental_box.isChecked()
        )

        loop_behavior = self._settings.loop_behavior
        loop_behavior.max_tool_rounds = _parse_int(
            self._max_tool_rounds_edit.text(), default=loop_behavior.max_tool_rounds
        )
        loop_behavior.idle_timeout_s = _parse_float(
            self._idle_timeout_edit.text(), default=loop_behavior.idle_timeout_s
        )
        loop_behavior.hard_timeout_s = _parse_float(
            self._hard_timeout_edit.text(), default=loop_behavior.hard_timeout_s
        )
        loop_behavior.budget_caution_remaining = _parse_int(
            self._budget_caution_remaining_edit.text(),
            default=loop_behavior.budget_caution_remaining,
        )
        loop_behavior.budget_critical_remaining = _parse_int(
            self._budget_critical_remaining_edit.text(),
            default=loop_behavior.budget_critical_remaining,
        )
        deadline_text = self._deadline_edit.text().strip()
        loop_behavior.deadline_s = (
            _parse_float(deadline_text, default=loop_behavior.deadline_s or 0.0)
            if deadline_text
            else None
        )
        loop_behavior.time_caution_remaining_s = _parse_float(
            self._time_caution_remaining_edit.text(),
            default=loop_behavior.time_caution_remaining_s,
        )
        loop_behavior.time_critical_remaining_s = _parse_float(
            self._time_critical_remaining_edit.text(),
            default=loop_behavior.time_critical_remaining_s,
        )

    def _on_save_clicked(self) -> None:
        settings = self.current_settings()
        if self._vault is not None:
            save_settings_with_vault(settings, self._settings_path, self._vault)
        else:
            settings.save(self._settings_path)
        self._settings = settings
        self.settings_saved.emit(settings)
        InfoBar.success(title="已保存", content="设置已保存", parent=self, duration=2000)


def _parse_int(text: str, default: int) -> int:
    try:
        return int(text.strip())
    except ValueError:
        return default


def _parse_float(text: str, default: float) -> float:
    try:
        return float(text.strip())
    except ValueError:
        return default


def _is_valid_int(text: str) -> bool:
    try:
        int(text.strip())
        return True
    except ValueError:
        return False


def _is_valid_float(text: str) -> bool:
    try:
        float(text.strip())
        return True
    except ValueError:
        return False


def _confirm_delete(parent: QWidget, name: str) -> bool:
    box = MessageBox("确认删除", f"确定要删除「{name}」吗？此操作不可撤销。", parent)
    return bool(box.exec())


class _McpServerListEditor(QWidget):
    """MCP server 列表的增删改：点击列表项把表单填充为该项数据，"新增/更新"按钮据是否
    选中列表项决定追加还是覆盖 ``configs`` 里的对应条目。
    """

    def __init__(self, configs: list[McpServerConfig], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._configs = configs

        self._list = ListWidget(self)
        _style_list_widget(self._list)
        self._name_edit = LineEdit(self)
        self._transport_combo = ComboBox(self)
        for transport in McpTransport:
            self._transport_combo.addItem(_MCP_TRANSPORT_LABELS[transport])
        self._transport_combo.currentIndexChanged.connect(self._on_transport_changed)
        self._command_edit = LineEdit(self)
        self._args_edit = LineEdit(self)
        self._env_edit = PlainTextEdit(self)
        self._url_edit = LineEdit(self)
        self._headers_edit = PlainTextEdit(self)
        self._enabled_box = CheckBox("启用", self)
        self._enabled_box.setChecked(True)
        self._trusted_box = CheckBox(
            "信任此 MCP server（豁免确认，但仍受路径沙箱/先读后改限制）", self
        )

        self._form = QFormLayout()
        self._form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._form.addRow("名称", self._name_edit)
        self._form.addRow("传输方式", self._transport_combo)
        self._form.addRow("命令", self._command_edit)
        self._form.addRow("参数（逗号分隔）", self._args_edit)
        self._form.addRow("环境变量（每行 KEY=VALUE）", self._env_edit)
        self._form.addRow("URL", self._url_edit)
        self._form.addRow("HTTP Header（每行 KEY=VALUE）", self._headers_edit)
        self._form.addRow(self._enabled_box)
        self._form.addRow(self._trusted_box)

        add_button = PushButton("新增/更新", self)
        remove_button = PushButton("删除", self)
        add_button.clicked.connect(self._on_add_or_update)
        remove_button.clicked.connect(self._on_remove)
        self._list.currentRowChanged.connect(self._on_selection_changed)

        button_row = QHBoxLayout()
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self._list)
        layout.addLayout(self._form)
        layout.addLayout(button_row)

        self._refresh_list()
        self._on_transport_changed(self._transport_combo.currentIndex())

    def _on_transport_changed(self, index: int) -> None:
        transport = list(McpTransport)[index]
        is_stdio = transport is McpTransport.STDIO
        self._form.setRowVisible(self._command_edit, is_stdio)
        self._form.setRowVisible(self._args_edit, is_stdio)
        self._form.setRowVisible(self._env_edit, is_stdio)
        self._form.setRowVisible(self._url_edit, not is_stdio)
        self._form.setRowVisible(self._headers_edit, not is_stdio)

    def _refresh_list(self) -> None:
        self._list.clear()
        for config in self._configs:
            suffix = "" if config.enabled else "（已禁用）"
            self._list.addItem(f"{config.name}{suffix}")

    def _on_selection_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._configs):
            return
        config = self._configs[row]
        self._name_edit.setText(config.name)
        self._transport_combo.setCurrentIndex(list(McpTransport).index(config.transport))
        self._command_edit.setText(config.command or "")
        self._args_edit.setText(_list_to_csv(config.args))
        self._env_edit.setPlainText(_env_dict_to_lines(config.env))
        self._url_edit.setText(config.url or "")
        self._headers_edit.setPlainText(_env_dict_to_lines(config.headers))
        self._enabled_box.setChecked(config.enabled)
        self._trusted_box.setChecked(config.trusted)

    def _on_add_or_update(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            return
        transport = list(McpTransport)[self._transport_combo.currentIndex()]
        if transport is McpTransport.STDIO:
            command = self._command_edit.text().strip()
            if not command:
                return
            config = McpServerConfig(
                name=name,
                transport=transport,
                command=command,
                args=_csv_to_list(self._args_edit.text()),
                env=_env_lines_to_dict(self._env_edit.toPlainText()),
                enabled=self._enabled_box.isChecked(),
                trusted=self._trusted_box.isChecked(),
            )
        else:
            url = self._url_edit.text().strip()
            if not url:
                return
            config = McpServerConfig(
                name=name,
                transport=transport,
                url=url,
                headers=_env_lines_to_dict(self._headers_edit.toPlainText()),
                enabled=self._enabled_box.isChecked(),
                trusted=self._trusted_box.isChecked(),
            )
        row = self._list.currentRow()
        if 0 <= row < len(self._configs):
            self._configs[row] = config
        else:
            self._configs.append(config)
        self._refresh_list()

    def _on_remove(self) -> None:
        row = self._list.currentRow()
        if not (0 <= row < len(self._configs)):
            return
        if not _confirm_delete(self, self._configs[row].name):
            return
        del self._configs[row]
        self._refresh_list()


class _AgentProfileListEditor(QWidget):
    """内部 sub-agent 画像列表的增删改，交互模式与 ``_McpServerListEditor`` 相同。"""

    def __init__(self, configs: list[AgentProfileConfig], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._configs = configs

        self._list = ListWidget(self)
        _style_list_widget(self._list)
        self._name_edit = LineEdit(self)
        self._system_prompt_edit = PlainTextEdit(self)
        self._enabled_box = CheckBox("启用", self)
        self._enabled_box.setChecked(True)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("名称", self._name_edit)
        form.addRow("系统提示词", self._system_prompt_edit)
        form.addRow(self._enabled_box)

        add_button = PushButton("新增/更新", self)
        remove_button = PushButton("删除", self)
        add_button.clicked.connect(self._on_add_or_update)
        remove_button.clicked.connect(self._on_remove)
        self._list.currentRowChanged.connect(self._on_selection_changed)

        button_row = QHBoxLayout()
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self._list)
        layout.addLayout(form)
        layout.addLayout(button_row)

        self._refresh_list()

    def _refresh_list(self) -> None:
        self._list.clear()
        for config in self._configs:
            suffix = "" if config.enabled else "（已禁用）"
            self._list.addItem(f"{config.name}{suffix}")

    def _on_selection_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._configs):
            return
        config = self._configs[row]
        self._name_edit.setText(config.name)
        self._system_prompt_edit.setPlainText(config.system_prompt)
        self._enabled_box.setChecked(config.enabled)

    def _on_add_or_update(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            return
        config = AgentProfileConfig(
            name=name,
            system_prompt=self._system_prompt_edit.toPlainText(),
            enabled=self._enabled_box.isChecked(),
        )
        row = self._list.currentRow()
        if 0 <= row < len(self._configs):
            self._configs[row] = config
        else:
            self._configs.append(config)
        self._refresh_list()

    def _on_remove(self) -> None:
        row = self._list.currentRow()
        if not (0 <= row < len(self._configs)):
            return
        if not _confirm_delete(self, self._configs[row].name):
            return
        del self._configs[row]
        self._refresh_list()


class _AcpAgentListEditor(QWidget):
    """ACP 外部 agent 列表的增删改，交互模式与 ``_McpServerListEditor`` 相同。"""

    def __init__(self, configs: list[AcpAgentConfig], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._configs = configs

        self._list = ListWidget(self)
        _style_list_widget(self._list)
        self._name_edit = LineEdit(self)
        self._executable_edit = LineEdit(self)
        self._args_edit = LineEdit(self)
        self._timeout_edit = LineEdit(self)
        self._timeout_edit.setPlaceholderText("留空则跟随全局默认")
        self._enabled_box = CheckBox("启用", self)
        self._enabled_box.setChecked(True)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("名称", self._name_edit)
        form.addRow("可执行文件路径", self._executable_edit)
        form.addRow("参数（逗号分隔）", self._args_edit)
        form.addRow("委派超时（秒）", self._timeout_edit)
        form.addRow(self._enabled_box)

        add_button = PushButton("新增/更新", self)
        remove_button = PushButton("删除", self)
        add_button.clicked.connect(self._on_add_or_update)
        remove_button.clicked.connect(self._on_remove)
        self._list.currentRowChanged.connect(self._on_selection_changed)

        button_row = QHBoxLayout()
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self._list)
        layout.addLayout(form)
        layout.addLayout(button_row)

        self._refresh_list()

    def _refresh_list(self) -> None:
        self._list.clear()
        for config in self._configs:
            suffix = "" if config.enabled else "（已禁用）"
            self._list.addItem(f"{config.name}{suffix}")

    def _on_selection_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._configs):
            return
        config = self._configs[row]
        self._name_edit.setText(config.name)
        self._executable_edit.setText(config.executable)
        self._args_edit.setText(_list_to_csv(config.args))
        self._timeout_edit.setText("" if config.timeout_s is None else str(config.timeout_s))
        self._enabled_box.setChecked(config.enabled)

    def _on_add_or_update(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            return
        timeout_text = self._timeout_edit.text().strip()
        timeout_s: float | None = None
        if timeout_text:
            try:
                timeout_s = float(timeout_text)
            except ValueError:
                timeout_s = None
        config = AcpAgentConfig(
            name=name,
            executable=self._executable_edit.text().strip(),
            args=_csv_to_list(self._args_edit.text()),
            enabled=self._enabled_box.isChecked(),
            timeout_s=timeout_s,
        )
        row = self._list.currentRow()
        if 0 <= row < len(self._configs):
            self._configs[row] = config
        else:
            self._configs.append(config)
        self._refresh_list()

    def _on_remove(self) -> None:
        row = self._list.currentRow()
        if not (0 <= row < len(self._configs)):
            return
        if not _confirm_delete(self, self._configs[row].name):
            return
        del self._configs[row]
        self._refresh_list()
