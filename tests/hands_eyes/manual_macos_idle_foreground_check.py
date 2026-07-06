"""主动交互感知能力的 macOS 端到端校验脚本：手动运行，验证生产代码而非一次性 spike。

不是 pytest 用例（文件名不匹配 test_*.py，不会被自动收集），因为 get_idle_seconds/
get_foreground_app_info 依赖真实的用户输入历史与前台窗口状态，自动化测试环境里没有
稳定可复现的期望值。用法::

    uv run python tests/hands_eyes/manual_macos_idle_foreground_check.py

预期效果：脚本打印当前空闲秒数（应接近 0，因为刚运行了脚本）与当前前台应用/窗口标题
（应与实际切到前台的窗口一致），供人工核对。
"""

from __future__ import annotations

import sys

from miku_on_desk.hands_eyes.backend import create_platform_backend


def main() -> int:
    backend = create_platform_backend()

    idle_seconds = backend.get_idle_seconds()
    print(f"距离上一次输入：{idle_seconds:.3f} 秒（预期接近 0，因为刚有键盘/鼠标活动）")

    app_info = backend.get_foreground_app_info()
    if app_info is None:
        print("get_foreground_app_info 返回 None——请确认当前有前台窗口后重试。")
        return 1

    print(f"前台应用：{app_info.app_name}，窗口标题：{app_info.window_title}")
    print("PASS：请人工核对以上应用名/窗口标题与当前实际前台窗口是否一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
