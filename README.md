# Miku On Desktop

一个初音未来桌面宠物 AI Agent：常驻在桌面上的透明悬浮窗精灵，可以随意拖动、会自己走来走去，
背后接一个 AI Agent 循环，能聊天、看屏幕截图、操作鼠标键盘、调用 MCP 工具、委派给
Claude Code / Codex 等外部 CLI Agent，并有一套跨会话的长期记忆系统。

## 功能特性

- **多 Provider 路由**：Anthropic / OpenAI / Gemini / Qwen(OpenAI 兼容),按 `mini/fast/medium/heavy`
  四档能力分层调度,可配置跨 Provider 降级。
- **桌宠 UI**：`PySide6` 无边框透明悬浮窗,支持拖拽、自主游走、多套角色立绘切换,对话气泡支持
  流式输出与是/否确认弹窗。
- **工具与 Agent**:
  - 内建工具:屏幕截图分析、鼠标键盘操作、情绪反应表达、记忆读写等。
  - MCP host:可接入任意本机(stdio)或远程(SSE / streamable-http) MCP server。
  - Skills:`Markdown + YAML frontmatter` 格式的技能包,支持热重载。
  - `spawn_agents`:内部 sub-agent 编排;`acp_delegate`:通过 Agent Client Protocol 委派给
    Claude Code / Codex 等外部 CLI Agent。
- **长期记忆**:四层文件系统记忆(语义 / 情景 / 情感 / 原始会话),带独立的记忆管理面板。
- **主动交互**:可选的定时/不定时主动搭话(基于屏幕内容 + 空闲时长 + 安静时段)。
- **Claude Code Hook 集成**:本地 hook sidecar 接收 Claude Code 等外部 CLI 工具的通知事件。

## 环境要求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)(依赖与虚拟环境管理)
- macOS 或 Windows(见下方平台专属依赖)

## 安装

```bash
git clone git@github.com:thunguo/miku-on-desktop.git
cd miku-on-desktop

# 按平台安装对应可选依赖
uv sync --extra macos      # macOS
uv sync --extra windows    # Windows
```

## 配置

配置分两层,存放位置不同:

1. **启动引导项**(`data_dir`/`log_level`):读环境变量或项目根目录下的 `.env` 文件,前缀
   `MIKU_`,例如:

   ```bash
   # .env(可选,不创建则全部使用默认值)
   MIKU_LOG_LEVEL=DEBUG
   MIKU_DATA_DIR=/custom/data/path
   ```

2. **应用配置**(各 Provider 的 API Key、MCP servers、Skills/Memory 目录、人格、快捷键等):
   不通过环境变量配置,而是首次启动后在托盘菜单的"设置"面板里填写,自动落盘为 JSON,路径由
   `platformdirs` 按操作系统决定,不在项目目录内、也不会被 git 追踪:

   | 平台 | 配置文件路径(示例) |
   | --- | --- |
   | macOS | `~/Library/Application Support/miku-on-desk/settings.json` |
   | Windows | `%LOCALAPPDATA%\miku-on-desk\miku-on-desk\settings.json` |

   首次运行若该文件不存在,会用全默认值创建一份(所有 Provider 默认禁用,需要在设置面板里
   填入至少一个 Provider 的 API Key 才能开始对话)。日志文件同样由 `platformdirs` 决定路径
   (macOS 下是 `~/Library/Logs/miku-on-desk`)。

## How to Run

```bash
uv run miku-on-desk
```

启动后会在系统托盘出现图标,桌面上出现悬浮宠物窗口。右键点击托盘图标或宠物本体可以打开
设置面板、记忆面板、角色画廊等。

## 开发

```bash
uv run ruff check .        # lint
uv run mypy src            # 类型检查(strict 模式,见 pyproject.toml)
uv run pytest -q           # 测试
```
