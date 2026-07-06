"""PetWalker 的回归测试：纯函数式随机水平游走——匀速趋近目标、到达后停顿、停顿结束后
换方向再走，以及全程不越界。用可注入的 `random.Random(seed)` 使目标/停顿时长可预测，
必要时用同一个 seed 起一个"回放" `Random` 实例重放同样顺序的调用来算出预期值，而不是
重复写死魔法数字。
"""

from __future__ import annotations

import random

from miku_on_desk.face.pet_motion import PetTargetWalker, PetWalker, compute_stand_position


def test_first_tick_does_not_move_and_only_records_time_baseline() -> None:
    walker = PetWalker(rng=random.Random(0))

    x = walker.tick(0.0, 500, (0, 1000))

    assert x == 500


def test_second_tick_jumps_exactly_to_random_target_when_speed_is_unbounded() -> None:
    seed = 42
    bounds = (0, 1000)
    walker = PetWalker(speed_px_per_s=1_000_000.0, rng=random.Random(seed))
    walker.tick(0.0, 500, bounds)

    replay = random.Random(seed)
    expected_target = replay.uniform(*bounds)

    x = walker.tick(1.0, 500, bounds)

    assert x == round(expected_target)


def test_walker_moves_toward_target_by_exactly_one_step_without_overshoot() -> None:
    seed = 7
    bounds = (0, 1000)
    walker = PetWalker(speed_px_per_s=10.0, rng=random.Random(seed))
    walker.tick(0.0, 500, bounds)

    replay = random.Random(seed)
    target = replay.uniform(*bounds)
    direction = 1 if target > 500 else -1

    x = walker.tick(1.0, 500, bounds)

    expected = 500 + direction * min(10.0, abs(target - 500))
    assert x == round(expected)
    assert (x - 500) * direction >= 0


def test_walker_pauses_immediately_after_reaching_target() -> None:
    seed = 3
    bounds = (0, 1000)
    walker = PetWalker(
        speed_px_per_s=1_000_000.0, pause_range_s=(5.0, 5.0), rng=random.Random(seed)
    )
    walker.tick(0.0, 500, bounds)
    x1 = walker.tick(1.0, 500, bounds)

    x2 = walker.tick(1.1, x1, bounds)

    assert x2 == x1


def test_walker_resumes_toward_new_target_after_pause_elapses() -> None:
    seed = 3
    bounds = (0, 1000)
    walker = PetWalker(
        speed_px_per_s=1_000_000.0, pause_range_s=(1.0, 1.0), rng=random.Random(seed)
    )
    walker.tick(0.0, 500, bounds)
    x1 = walker.tick(1.0, 500, bounds)
    x_paused = walker.tick(1.5, x1, bounds)
    assert x_paused == x1

    x_resumed = walker.tick(2.5, x1, bounds)

    replay = random.Random(seed)
    replay.uniform(*bounds)
    replay.uniform(1.0, 1.0)
    expected_target2 = replay.uniform(*bounds)
    assert x_resumed == round(expected_target2)


def test_facing_right_true_when_target_bounds_entirely_to_the_right() -> None:
    walker = PetWalker(rng=random.Random(1))
    walker.tick(0.0, 0, (500, 1500))

    walker.tick(1.0, 0, (500, 1500))

    assert walker.facing_right is True


def test_facing_right_false_when_target_bounds_entirely_to_the_left() -> None:
    walker = PetWalker(rng=random.Random(1))
    walker.tick(0.0, 2000, (500, 1500))

    walker.tick(1.0, 2000, (500, 1500))

    assert walker.facing_right is False


def test_position_never_leaves_bounds_over_many_ticks() -> None:
    bounds = (100, 400)
    walker = PetWalker(speed_px_per_s=37.0, pause_range_s=(0.05, 0.2), rng=random.Random(123))
    x = walker.tick(0.0, 250, bounds)
    t = 0.0
    for _ in range(500):
        t += 0.033
        x = walker.tick(t, x, bounds)
        assert bounds[0] <= x <= bounds[1]


def test_degenerate_bounds_pins_target_to_min_x() -> None:
    walker = PetWalker(speed_px_per_s=1000.0, rng=random.Random(5))
    walker.tick(0.0, 300, (200, 200))

    x = walker.tick(1.0, 300, (200, 200))

    assert x == 200


_SCREEN_RECT = (0, 0, 2000, 1000)
_SPRITE_W, _SPRITE_H = 100, 150


def test_stand_position_left_candidate_never_overlaps_target() -> None:
    stand_x, _ = compute_stand_position(1000, 500, _SPRITE_W, _SPRITE_H, _SCREEN_RECT, 0)

    assert stand_x + _SPRITE_W <= 1000


def test_stand_position_right_candidate_never_overlaps_target() -> None:
    stand_x, _ = compute_stand_position(1000, 500, _SPRITE_W, _SPRITE_H, _SCREEN_RECT, 2000)

    assert stand_x >= 1000


def test_stand_position_prefers_side_closer_to_current_x() -> None:
    left_x, _ = compute_stand_position(1000, 500, _SPRITE_W, _SPRITE_H, _SCREEN_RECT, 0)
    right_x, _ = compute_stand_position(1000, 500, _SPRITE_W, _SPRITE_H, _SCREEN_RECT, 2000)

    assert left_x < 1000 < right_x


def test_stand_position_exact_tie_prefers_right_deterministically() -> None:
    stand_x, _ = compute_stand_position(1000, 500, _SPRITE_W, _SPRITE_H, _SCREEN_RECT, 950)

    assert stand_x == 1024


def test_stand_position_y_centers_on_target_when_away_from_edges() -> None:
    _, stand_y = compute_stand_position(1000, 500, _SPRITE_W, _SPRITE_H, _SCREEN_RECT, 950)

    assert stand_y == 425


def test_stand_position_y_clamps_near_top_edge() -> None:
    _, stand_y = compute_stand_position(1000, 10, _SPRITE_W, _SPRITE_H, _SCREEN_RECT, 950)

    assert stand_y == 0


def test_stand_position_y_clamps_near_bottom_edge() -> None:
    _, stand_y = compute_stand_position(1000, 990, _SPRITE_W, _SPRITE_H, _SCREEN_RECT, 950)

    assert stand_y == 850


def test_stand_position_falls_back_to_right_when_left_side_infeasible() -> None:
    stand_x, _ = compute_stand_position(10, 500, _SPRITE_W, _SPRITE_H, _SCREEN_RECT, 2000)

    assert stand_x == 34
    assert stand_x >= 10


def test_stand_position_falls_back_to_left_when_right_side_infeasible() -> None:
    stand_x, _ = compute_stand_position(1990, 500, _SPRITE_W, _SPRITE_H, _SCREEN_RECT, 0)

    assert stand_x == 1866
    assert stand_x + _SPRITE_W <= 1990


def test_stand_position_degenerate_narrow_screen_does_not_raise() -> None:
    stand_x, stand_y = compute_stand_position(75, 500, _SPRITE_W, _SPRITE_H, (0, 0, 150, 1000), 0)

    assert isinstance(stand_x, int)
    assert isinstance(stand_y, int)
    assert stand_x == 50


def test_target_walker_first_tick_only_records_baseline_and_does_not_move() -> None:
    walker = PetTargetWalker()

    x, y = walker.tick(0.0, 100, 100, (500, 500))

    assert (x, y) == (100, 100)


def test_target_walker_moves_by_speed_times_dt_without_overshoot() -> None:
    walker = PetTargetWalker(speed_px_per_s=10.0)
    walker.tick(0.0, 0, 0, (100, 0))

    x, y = walker.tick(1.0, 0, 0, (100, 0))

    assert (x, y) == (10, 0)


def test_target_walker_snaps_to_target_when_within_one_step() -> None:
    walker = PetTargetWalker(speed_px_per_s=1000.0)
    walker.tick(0.0, 0, 0, (10, 10))

    x, y = walker.tick(1.0, 0, 0, (10, 10))

    assert (x, y) == (10, 10)


def test_target_walker_stays_put_once_arrived() -> None:
    walker = PetTargetWalker(speed_px_per_s=1000.0)
    walker.tick(0.0, 0, 0, (10, 10))
    x, y = walker.tick(1.0, 0, 0, (10, 10))

    x2, y2 = walker.tick(2.0, x, y, (10, 10))

    assert (x2, y2) == (10, 10)


def test_target_walker_reset_prevents_teleport_from_stale_time_baseline() -> None:
    walker = PetTargetWalker(speed_px_per_s=10.0)
    walker.tick(0.0, 0, 0, (1000, 0))

    walker.reset()
    x, y = walker.tick(500.0, 5, 5, (1000, 0))

    assert (x, y) == (5, 5)
