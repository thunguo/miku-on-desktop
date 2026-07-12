# Miku On Desktop [![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=for-the-badge&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/thunguo/miku-on-desktop)

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

### Raspberry Pi 直接渲染 kiosk

树莓派 MHS-3.5 屏幕可不启动 Linux 桌面，直接把 kiosk 渲染到 `/dev/fb0`。部署前先在
树莓派完成依赖安装，并确认 `miku-on-desk-kiosk` 能在桌面会话中正常启动：

```bash
cd /home/thunguo/miku-on-desk
uv sync
sudo ./scripts/install-kiosk-direct-render.sh
```

脚本会停止并禁用 LightDM/X11，安装 `miku-kiosk.service`，然后以 Qt LinuxFB 插件直接渲染
到屏幕。启动或运行异常不会回到 Linux 桌面：已初始化 Qt 时会显示 Miku 的错误页，详细日志在：

```bash
journalctl -u miku-kiosk.service -f
```

通过 SSH 恢复原桌面：

```bash
sudo /home/thunguo/miku-on-desk/scripts/restore-desktop.sh
```

## 开发

```bash
uv run ruff check .        # lint
uv run mypy src            # 类型检查(strict 模式,见 pyproject.toml)
uv run pytest -q           # 测试
```

### macOS 摄像头/麦克风权限(克隆角色功能调试)

"克隆角色"流程需要拍照(摄像头)和录音(麦克风)。开发时通过 `uv run miku-on-desk` /
IDE 直接运行,实际跑起来的是 uv 管理的裸 `python3` 解释器,它自己没有 `Info.plist`,
不能声明 `NSCameraUsageDescription`/`NSMicrophoneUsageDescription`。macOS 会把权限请求
归因给"责任进程"——沿父进程链向上找到的第一个已签名、声明了对应用途说明的 App,
一般是你用来启动它的终端或 IDE(PyCharm / Terminal / iTerm2 / VS Code 等),**不是**
`miku-on-desk` 自己。

首次进入拍照/录音步骤时:

1. 系统会弹出权限对话框,写的是你的终端/IDE 的名字(例如"PyCharm 想访问相机"),
   点"允许"即可,之后同一个终端/IDE 启动的进程都会沿用这个授权。
2. 如果没有弹窗、直接提示"权限被拒绝":说明之前已经有一次被记成了拒绝(常见于权限
   请求过程中程序崩溃导致的异常状态)。用 `tccutil` 清掉缓存的决定后重试:

   ```bash
   tccutil reset Camera <bundle-id>
   tccutil reset Microphone <bundle-id>
   ```

   常见 `<bundle-id>`:PyCharm 是 `com.jetbrains.pycharm`,Terminal.app 是
   `com.apple.Terminal`,iTerm2 是 `com.googlecode.iterm2`,VS Code 是
   `com.microsoft.VSCode`。执行后**完全退出并重新打开**该终端/IDE(不能只是重跑脚本),
   再重新走一次拍照/录音流程。
3. 也可以在"系统设置 → 隐私与安全性 → 摄像头/麦克风"里直接查看、手动开关对应
   终端/IDE 的授权状态。
