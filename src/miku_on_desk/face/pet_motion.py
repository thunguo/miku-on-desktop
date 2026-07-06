"""桌宠窗口的两种自主移动逻辑：空闲游走（`PetWalker`）与"走到操作目标附近"（`PetTargetWalker`）。

与 `pet_state.py` 的 `PetStateMachine` 同级、同风格：外部单调时钟 `t` 参数化，`tick()`
在概念上是纯函数（内部只记进度，不缓存"当前在哪"——每次调用都以调用方传入的当前坐标为
准）。是否该调用 `tick()`（例如只在 IDLE 态才走）完全由调用方（`overlay_window.py`）决定，
这里不知道状态机、窗口或 Qt 的存在。

`PetWalker`（随机选一个目标 x → 匀速走过去 → 停顿一段随机时长 → 再选下一个目标）：
可注入 `rng` 使测试确定化。`current_x` 必须以调用方为准而非自行记账：`_on_animation_tick`
只在 IDLE 态才调用 `tick()`，拖拽等其他状态期间窗口位置可能被外部改变，恢复行走时必须从
窗口实际所在位置继续，而不是从游走内部记的、可能早已过时的旧位置突然跳过去。停顿结束、
真正开始移动的那一刻才重新选目标、更新 `facing_right`，而不是一到达旧目标就选好下一个——
这样转身发生在实际迈步的瞬间，而不是还站着不动就先转了身。

`compute_stand_position`/`PetTargetWalker`：AI 即将点击屏幕上某个坐标时，让宠物窗口
直线走到那附近但不遮挡该点。`compute_stand_position` 只负责算"该站哪"（一次性几何计算，
不含时间维度）；`PetTargetWalker` 只负责"怎么走过去"（按固定速度逐帧逼近一个给定目标点），
与 `PetWalker` 一样不知道这个目标点是怎么来的。
"""

from __future__ import annotations

import random

_DEFAULT_SPEED_PX_PER_S = 40.0
_DEFAULT_PAUSE_RANGE_S = (2.0, 6.0)


class PetWalker:
    """在一维水平区间内随机游走的纯逻辑状态机，不依赖 Qt/窗口/状态机。"""

    def __init__(
        self,
        *,
        speed_px_per_s: float = _DEFAULT_SPEED_PX_PER_S,
        pause_range_s: tuple[float, float] = _DEFAULT_PAUSE_RANGE_S,
        rng: random.Random | None = None,
    ) -> None:
        self._speed_px_per_s = speed_px_per_s
        self._pause_range_s = pause_range_s
        self._rng = rng if rng is not None else random.Random()
        self._target_x = 0.0
        self._facing_right = True
        self._resume_at = 0.0
        self._need_new_target = False
        self._last_t: float | None = None

    @property
    def facing_right(self) -> bool:
        return self._facing_right

    def tick(self, t: float, current_x: int, bounds: tuple[int, int]) -> int:
        """返回这一 tick 后应处于的绝对 x（已 clamp 进 `bounds`）。

        首次调用只记录时间基准、不移动，避免窗口一启动就跳到随机位置。
        """
        if self._last_t is None:
            self._last_t = t
            self._need_new_target = True
            self._resume_at = t
            return round(self._clamp(current_x, bounds))

        dt = max(0.0, t - self._last_t)
        self._last_t = t

        if t < self._resume_at:
            return round(self._clamp(current_x, bounds))

        if self._need_new_target:
            self._pick_new_target(current_x, bounds)
            self._need_new_target = False

        step = self._speed_px_per_s * dt
        x = float(current_x)
        if x < self._target_x:
            x = min(x + step, self._target_x)
        elif x > self._target_x:
            x = max(x - step, self._target_x)

        if x == self._target_x:
            self._resume_at = t + self._rng.uniform(*self._pause_range_s)
            self._need_new_target = True

        return round(self._clamp(x, bounds))

    def _pick_new_target(self, current_x: int, bounds: tuple[int, int]) -> None:
        min_x, max_x = bounds
        self._target_x = float(min_x) if min_x >= max_x else self._rng.uniform(min_x, max_x)
        if self._target_x > current_x:
            self._facing_right = True
        elif self._target_x < current_x:
            self._facing_right = False

    @staticmethod
    def _clamp(x: float, bounds: tuple[int, int]) -> float:
        min_x, max_x = bounds
        if min_x > max_x:
            return float(min_x)
        return max(float(min_x), min(float(max_x), x))


_DEFAULT_HURRY_SPEED_PX_PER_S = 120.0  # 营造"赶过去"的观感
_DEFAULT_TARGET_MARGIN_PX = 24.0  # 窗口边缘与目标点之间预留的间隙


def _clamp(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        return lo
    return max(lo, min(hi, x))


def compute_stand_position(
    target_x: int,
    target_y: int,
    sprite_width: int,
    sprite_height: int,
    screen_rect: tuple[int, int, int, int],
    current_x: int,
    margin: float = _DEFAULT_TARGET_MARGIN_PX,
) -> tuple[int, int]:
    """算出"离目标点近、但窗口整个横向范围都不覆盖目标点"的站位。

    几何依据：点被矩形包含要求 x/y 两个区间同时命中；只要保证窗口的横向区间与
    target_x（留 margin）完全不相交，无论 y 选哪个值，目标点都不可能落在窗口
    矩形内——因此 y 可以完全自由地选（贴近 target_y，方便"看起来站得近"），
    不需要复杂的 2D 碰撞判断。
    """
    left, top, right, bottom = screen_rect
    min_win_x, max_win_x = float(left), float(right - sprite_width)
    min_win_y, max_win_y = float(top), float(bottom - sprite_height)

    x_right = _clamp(target_x + margin, min_win_x, max_win_x)
    x_left = _clamp(target_x - margin - sprite_width, min_win_x, max_win_x)
    feasible_right = x_right > target_x
    feasible_left = (x_left + sprite_width) < target_x

    candidates = [
        (x, abs(current_x - x))
        for x, feasible in [(x_right, feasible_right), (x_left, feasible_left)]
        if feasible
    ]
    if candidates:
        candidates.sort(key=lambda c: (c[1], c[0] != x_right))  # 就近；平手偏右（确定性）
        stand_x = candidates[0][0]
    else:
        # 屏幕窄到两侧都放不下整个 sprite + margin 时的退化兜底：选间隙较大的一侧，
        # 保证函数总有合理返回值而不是抛异常。
        gap_right = x_right - target_x
        gap_left = target_x - (x_left + sprite_width)
        stand_x = x_right if gap_right >= gap_left else x_left

    stand_y = _clamp(target_y - sprite_height / 2, min_win_y, max_win_y)
    return round(stand_x), round(stand_y)


class PetTargetWalker:
    """朝一个目标点直线趋近，速度快于空闲游走。与 PetWalker 同风格：外部时钟参数化，
    调用方每帧传入当前位置，本类只记录时间基准与目标点，不缓存"当前在哪"。
    """

    def __init__(self, *, speed_px_per_s: float = _DEFAULT_HURRY_SPEED_PX_PER_S) -> None:
        self._speed_px_per_s = speed_px_per_s
        self._last_t: float | None = None

    def tick(
        self, t: float, current_x: int, current_y: int, target: tuple[int, int]
    ) -> tuple[int, int]:
        """首次调用（或 reset() 后的首次调用）只记录时间基准、不移动，避免用一个
        过期的 _last_t 算出巨大 dt 导致瞬移到目标点。
        """
        if self._last_t is None:
            self._last_t = t
            return current_x, current_y

        dt = max(0.0, t - self._last_t)
        self._last_t = t
        target_x, target_y = target
        dx, dy = target_x - current_x, target_y - current_y
        distance = (dx * dx + dy * dy) ** 0.5
        step = self._speed_px_per_s * dt
        if distance <= step or distance == 0.0:
            return target_x, target_y
        ratio = step / distance
        return round(current_x + dx * ratio), round(current_y + dy * ratio)

    def reset(self) -> None:
        """开始追一个新目标前调用，清空时间基准——避免长时间未 tick 后突然算出巨大 dt。"""
        self._last_t = None
