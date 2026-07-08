"""共享的指数退避辅助函数：从 ``brain/providers/retry.py`` 抽出来，供跨模块的"失败后重试"
场景复用（Provider 流式重试、MCP 连接自愈看门狗），不必各自重新发明一套抖动+封顶算法。
"""

from __future__ import annotations

import random

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY_S = 1.0
DEFAULT_MAX_DELAY_S = 20.0


def backoff_delay(attempt: int, *, base_delay_s: float, max_delay_s: float) -> float:
    """指数退避延迟：``base_delay_s * 2**attempt``，叠加 0.5x-1.5x 抖动后按
    ``max_delay_s`` 封顶。
    """
    exponential = base_delay_s * (2**attempt)
    jittered = exponential * (0.5 + random.random())
    return float(min(jittered, max_delay_s))
