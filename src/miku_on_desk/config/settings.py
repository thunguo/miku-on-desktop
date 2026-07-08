"""集中式配置模型：Provider 凭证/路由、MCP/Skills/Agents/ACP 列表、窗口与日志设置。

配置分两层：
- ``EnvBootstrap`` 只负责启动阶段从环境变量/``.env`` 读取的极少量引导项（数据目录覆盖、日志级别），
  这些项在进程运行期间不会被 UI 改写。
- ``AppSettings`` 是可被设置面板在运行时读写的完整配置树，落盘为用户配置目录下的 JSON 文件，
  而不是环境变量——因为 GUI 编辑的配置需要能被程序写回，环境变量做不到这一点。
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from platformdirs import PlatformDirs
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from miku_on_desk.brain.secrets.vault import SecretVault

_APP_NAME = "miku-on-desk"
_APP_AUTHOR = "miku-on-desk"

_dirs = PlatformDirs(appname=_APP_NAME, appauthor=_APP_AUTHOR)


class ModelTier(StrEnum):
    """模型分层路由的层级：按能力/成本从低到高。"""

    MINI = "mini"
    FAST = "fast"
    MEDIUM = "medium"
    HEAVY = "heavy"


class ProviderName(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    QWEN = "qwen"


class EnvBootstrap(BaseSettings):
    """进程启动时从环境变量/.env 读取的引导配置，不通过 UI 修改。"""

    model_config = SettingsConfigDict(env_prefix="MIKU_", env_file=".env", extra="ignore")

    data_dir: Path | None = None
    log_level: str = "INFO"

    def resolve_data_dir(self) -> Path:
        return self.data_dir if self.data_dir is not None else Path(_dirs.user_data_dir)

    def resolve_config_dir(self) -> Path:
        return Path(_dirs.user_config_dir)

    def resolve_log_dir(self) -> Path:
        return Path(_dirs.user_log_dir)


class ProviderConfig(BaseModel):
    """单个 LLM Provider 的凭证与分层模型名，均可留空表示未启用该 Provider。"""

    api_key: str | None = None
    base_url: str | None = None
    models: dict[ModelTier, str] = Field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.api_key is not None and bool(self.models)


class ModelRouterConfig(BaseModel):
    """四个 Provider 的配置集合；model_router 只从其中 enabled=True 的 Provider 里组装路由表。"""

    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)
    qwen: ProviderConfig = Field(default_factory=ProviderConfig)
    enable_cross_provider_fallback: bool = False
    """当前 Provider 重试耗尽后，是否允许降级到另一个已启用的同等层级 Provider。默认关闭：
    不同 Provider 的模型行为/系统提示适配程度不同，静默切换可能改变回复风格，需用户主动选择。
    """

    def provider(self, name: ProviderName) -> ProviderConfig:
        return getattr(self, name.value)  # type: ignore[no-any-return]

    def enabled_providers(self) -> list[ProviderName]:
        return [name for name in ProviderName if self.provider(name).enabled]


class McpTransport(StrEnum):
    """MCP server 的连接方式：本机子进程，或远程 HTTP server。

    ``STREAMABLE_HTTP`` 的值用连字符而非下划线，以匹配
    ``mcp.server.fastmcp.FastMCP.run(transport=...)`` 接受的字面量，避免测试/装配代码里
    还要做一次值转换。
    """

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable-http"


class McpServerConfig(BaseModel):
    """一个外部 MCP server 的连接方式：stdio 启动本机子进程，或 sse/streamable-http 连接
    远程 server（远程两种都可配置自定义 HTTP header，通常用于鉴权）。

    ``command``/``args``/``env`` 仅 stdio 使用，``url``/``headers`` 仅远程两种 transport
    使用——不加跨字段校验强制二选一，与本文件其余配置类的风格一致（校验交给 UI 层做
    基本的必填检查，配置本身允许"暂时不完整"）。
    """

    name: str
    transport: McpTransport = McpTransport.STDIO
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    trusted: bool = False
    """用户显式标记为可信时，豁免该 server 桥接工具的 `requires_confirmation`——但不豁免
    路径沙箱/先读后改这两条结构性边界，见 `brain/mcp/host.py::_infer_policy_spec`。"""


class AgentProfileConfig(BaseModel):
    """内部 sub-agent 画像（researcher/operator/planner 等），供 spawn_agents 编排使用。"""

    name: str
    system_prompt: str
    enabled: bool = True


class AcpAgentConfig(BaseModel):
    """通过 ACP（Agent Client Protocol）调用的本机外部 agent，如 Claude Code、Codex。"""

    name: str
    executable: str
    args: list[str] = Field(default_factory=list)
    enabled: bool = True
    timeout_s: float | None = None
    """覆盖单个 agent 的委派超时；None 回退到 ``LongTaskConfig.acp_delegate_default_timeout_s``"""


class LongTaskConfig(BaseModel):
    """`spawn_agents`/`acp_delegate` 这类可能跑数分钟到十几分钟的委派工具的墙钟超时设置。"""

    spawn_agents_deadline_s: float = 600.0
    acp_delegate_default_timeout_s: float = 900.0


class LoopBehaviorConfig(BaseModel):
    """``brain/loop.py::LoopConfig`` 的用户可配置镜像，字段含义与默认值与其一一对应。"""

    max_tool_rounds: int = 100
    idle_timeout_s: float = 120.0
    hard_timeout_s: float = 600.0
    budget_caution_remaining: int = 10
    budget_critical_remaining: int = 3
    deadline_s: float | None = None
    time_caution_remaining_s: float = 60.0
    time_critical_remaining_s: float = 20.0


class MemoryTuningConfig(BaseModel):
    """记忆检索/整理/屏幕匹配相关阈值，默认值与原硬编码常量一致。

    多数字段目前只被生产代码里已有的活跃调用链（compaction/screen_analyze）消费；
    ``retrieval_min_confidence``/``base_similarity_threshold``/
    ``emotional_confidence_threshold`` 对应的方法当前没有生产调用点（仅测试直接调用），
    先做成可配置项，供未来接入真实调用路径时复用。
    """

    retrieval_min_confidence: float = 0.7
    base_similarity_threshold: float = 0.80
    emotional_confidence_threshold: float = 0.75
    compaction_token_threshold: int = 60_000
    compaction_keep_recent: int = 6
    screen_match_threshold: float = 0.6


class PermissionsConfig(BaseModel):
    """``brain/tools/policy.py`` 的信任层与结构性边界依赖的用户可配置项。

    ``allowed_tools``/``denied_tools`` 只影响信任层（把本该询问的情形提升为直接放行，或彻底
    禁用），不能豁免路径沙箱与先读后改这两条结构性检查——原因见 ``policy.py`` 模块文档。
    """

    trusted_mode: bool = False
    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    allowed_dirs: list[Path] = Field(default_factory=list)
    default_decision: Literal["ask", "deny"] = "ask"


class WindowConfig(BaseModel):
    """桌宠悬浮窗的位置与显示设置。"""

    x: int = 100
    y: int = 100
    scale: float = 1.0
    always_on_top: bool = True
    walk_enabled: bool = True
    pet_dir: Path | None = None


class ShortcutsConfig(BaseModel):
    """确认对话框（是/否）的键盘快捷键，默认在 mac/Windows 上是同一个物理键位组合。"""

    confirm_yes: str = "Ctrl+Shift+Y"
    confirm_no: str = "Ctrl+Shift+N"


class PersonaConfig(BaseModel):
    """Miku 的人格：结构化字段，能力契约（工具列表/确认授权规则）不在此列，代码里固定。"""

    name: str = "初音未来"
    role: str = "寄居在用户电脑桌面上的虚拟伙伴"
    personality: str = "简短、口语化、带点活泼和小任性，不说教、不长篇大论、不机械罗列步骤"


class ProactiveConfig(BaseModel):
    """主动交互：定时/不定时地根据屏幕内容主动搭话。改动后需重启应用生效，
    与其余设置项（如 persona/model_router）的既有行为一致，不做热重载。"""

    enabled: bool = False
    min_interval_s: int = 600
    max_interval_s: int = 1800
    idle_threshold_s: int = 120
    quiet_hours_start: str | None = None
    quiet_hours_end: str | None = None
    max_daily_triggers: int = 10


class ComputerUseConfig(BaseModel):
    """`computer_input` 工具的自动结算与焦点漂移检测闭环。默认关闭：涉及工具执行后自动
    介入这一产品行为变化，需要用户主动打开。"""

    enabled: bool = False
    settle_delay_s: float = 0.3


class HookServerConfig(BaseModel):
    """本地 hook sidecar（接收 Claude Code 等外部 CLI 工具的通知）的监听设置。

    ``include_experimental`` 默认关闭：见 ``face/hooks/installer.py`` 模块文档,
    ``PreToolUse``/``PermissionRequest``/``PermissionDenied`` 的响应体可能被 Claude Code
    当作真正的允许/拒绝决策使用，启用前必须重新核实当时最新的官方文档。
    """

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765
    include_experimental: bool = False


class ImageGenerationConfig(BaseModel):
    """角色生成对话框使用的图像生成 API 凭证，跨会话复用避免重复输入。"""

    api_key: str | None = None
    base_url: str | None = None
    model: str = "gpt-image-1"


class AppSettings(BaseModel):
    """完整可配置项树，由设置面板读写，落盘为 JSON。"""

    model_router: ModelRouterConfig = Field(default_factory=ModelRouterConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)
    skills_dir: Path | None = None
    memory_dir: Path | None = None
    agent_profiles: list[AgentProfileConfig] = Field(default_factory=list)
    acp_agents: list[AcpAgentConfig] = Field(default_factory=list)
    window: WindowConfig = Field(default_factory=WindowConfig)
    hook_server: HookServerConfig = Field(default_factory=HookServerConfig)
    image_generation: ImageGenerationConfig = Field(default_factory=ImageGenerationConfig)
    shortcuts: ShortcutsConfig = Field(default_factory=ShortcutsConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    proactive: ProactiveConfig = Field(default_factory=ProactiveConfig)
    long_tasks: LongTaskConfig = Field(default_factory=LongTaskConfig)
    loop_behavior: LoopBehaviorConfig = Field(default_factory=LoopBehaviorConfig)
    memory_tuning: MemoryTuningConfig = Field(default_factory=MemoryTuningConfig)
    computer_use: ComputerUseConfig = Field(default_factory=ComputerUseConfig)

    @classmethod
    def load(cls, path: Path) -> AppSettings:
        if not path.exists():
            return cls()
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


def default_settings_path(bootstrap: EnvBootstrap | None = None) -> Path:
    bootstrap = bootstrap or EnvBootstrap()
    return bootstrap.resolve_config_dir() / "settings.json"


_VAULT_REF_PREFIX = "vault-ref:"
_IMAGE_GENERATION_VAULT_KEY = "image_generation_api_key"


def _provider_vault_key(name: ProviderName) -> str:
    return f"provider_api_key:{name.value}"


def _vault_ref(vault_key: str) -> str:
    return f"{_VAULT_REF_PREFIX}{vault_key}"


def _migrate_or_resolve(
    value: str | None, vault_key: str, vault: SecretVault
) -> tuple[str | None, bool]:
    """返回 `(明文值, 是否是本次触发迁移的遗留明文)`。

    命中 vault-ref 前缀时直接解密返回；否则视为旧版明文配置，立即存入 vault 并把
    第二个返回值置 True，提示调用方磁盘上的引用尚未落地，需要触发一次
    ``save_settings_with_vault`` 把明文改写成引用。
    """
    if value is None:
        return None, False
    if value.startswith(_VAULT_REF_PREFIX):
        return vault.get(value.removeprefix(_VAULT_REF_PREFIX)), False
    vault.store(vault_key, value)
    return value, True


def load_settings_with_vault(path: Path, vault: SecretVault) -> AppSettings:
    """加载 settings，并把发现的旧版明文 api_key 自动迁移进 vault、磁盘上改写为引用。

    返回值里的 api_key 字段始终是解密后的明文，供调用方（如 Provider 构造）直接使用；
    磁盘上则只留 ``vault-ref:<key>`` 引用，不会因为这次加载而新增明文。
    """
    settings = AppSettings.load(path)
    migrated = False

    for name in ProviderName:
        provider = settings.model_router.provider(name)
        provider.api_key, was_migrated = _migrate_or_resolve(
            provider.api_key, _provider_vault_key(name), vault
        )
        migrated = migrated or was_migrated

    settings.image_generation.api_key, image_migrated = _migrate_or_resolve(
        settings.image_generation.api_key, _IMAGE_GENERATION_VAULT_KEY, vault
    )
    migrated = migrated or image_migrated

    if migrated:
        save_settings_with_vault(settings, path, vault)

    return settings


def save_settings_with_vault(settings: AppSettings, path: Path, vault: SecretVault) -> None:
    """把 settings 存盘：api_key 字段先存入 vault，磁盘上只留引用。

    在 `settings` 的深拷贝上操作，不修改调用方持有的原始对象——调用方的内存副本需要
    继续持有明文，供运行时鉴权使用。
    """
    disk_copy = settings.model_copy(deep=True)

    for name in ProviderName:
        provider = disk_copy.model_router.provider(name)
        if provider.api_key is not None and not provider.api_key.startswith(_VAULT_REF_PREFIX):
            vault_key = _provider_vault_key(name)
            vault.store(vault_key, provider.api_key)
            provider.api_key = _vault_ref(vault_key)

    image_key = disk_copy.image_generation.api_key
    if image_key is not None and not image_key.startswith(_VAULT_REF_PREFIX):
        vault.store(_IMAGE_GENERATION_VAULT_KEY, image_key)
        disk_copy.image_generation.api_key = _vault_ref(_IMAGE_GENERATION_VAULT_KEY)

    disk_copy.save(path)
