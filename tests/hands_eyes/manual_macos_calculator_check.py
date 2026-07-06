"""Phase 1 macOS 端到端校验脚本：手动运行，验证生产代码而非 /tmp 里的一次性 spike。

不是 pytest 用例（文件名不匹配 test_*.py，不会被自动收集），因为它会真实点击系统
Calculator 应用、依赖辅助功能权限已授权、且会改变前台窗口，不适合在无人值守的
测试/CI 流程里跑。用法::

    uv run python tests/hands_eyes/manual_macos_calculator_check.py

预期效果：Calculator 前台弹出，数字键 7 被真实点击，脚本打印通过/失败结论。
"""

from __future__ import annotations

import subprocess
import sys
import time

from miku_on_desk.hands_eyes.backend import create_platform_backend
from miku_on_desk.hands_eyes.macos.accessibility import is_accessibility_trusted


def _find_calculator_pid() -> int:
    output = subprocess.check_output(
        [
            "osascript",
            "-e",
            'tell application "System Events" to unix id of process "Calculator"',
        ]
    )
    return int(output.decode().strip())


def main() -> int:
    if not is_accessibility_trusted():
        print("辅助功能权限未授权（AXIsProcessTrusted 为 False），请在系统设置里授权后重试。")
        return 1

    subprocess.run(["osascript", "-e", 'tell application "Calculator" to activate'], check=True)
    time.sleep(1)

    pid = _find_calculator_pid()
    backend = create_platform_backend()
    elements = backend.list_elements(pid)
    if not elements:
        print("list_elements 返回空列表，accessibility tree 枚举失败。")
        return 1

    seven = next((e for e in elements if e.label == "7"), None)
    if seven is None:
        labels = sorted({e.label for e in elements})
        print(f"未找到标签为 '7' 的按钮，实际拿到的标签集合：{labels}")
        return 1

    print(
        f"找到按钮 '7'：中心坐标=({seven.center_x}, {seven.center_y})，"
        f"尺寸=({seven.width}x{seven.height})"
    )
    backend.click(seven.center_x, seven.center_y)
    time.sleep(0.3)

    display_elements = backend.list_elements(pid)
    display = next((e for e in display_elements if e.role == "AXStaticText"), None)
    if display is not None:
        print(f"点击后显示区域文本：{display.label!r}")

    print("PASS：production 代码路径（backend -> accessibility -> pynput）端到端跑通。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
