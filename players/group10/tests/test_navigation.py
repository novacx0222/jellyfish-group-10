"""Tests for Group 10 headings, route planning, holding and search.

Run with: uv run python -m pytest players/group10/tests
"""

import random
from math import hypot

import pytest

from core.depth import Depth
from core.parse_args import parse_args
from core.runner import build_engine
from players.group10 import navigation as nav

# Four currents spanning the plane.
SPREAD = [nav.velocity(d, h) for d, h in zip(nav.DEPTHS, (45, 135, 225, 315))]
# Every current heads roughly north: most goals need a trip around the torus.
NORTHISH = [nav.velocity(d, h) for d, h in zip(nav.DEPTHS, (344, 341, 20, 31))]


@pytest.mark.parametrize("memory_size", [8, 15, 16, 1024])
def test_heading_store_round_trip(memory_size):
    from players.group10 import dead_reckoning as dr

    layout = dr.layout_for(memory_size)
    store = nav.heading_store_for(memory_size, layout.num_bytes)
    memory = bytearray(memory_size)
    assert store.read(memory) == [None] * 4
    for d, h in zip(nav.DEPTHS, (0.0, 97.0, 200.0, 359.0)):
        store.write(memory, d, h)
    step = 360 / store.sectors
    for got, want in zip(store.read(memory), (0.0, 97.0, 200.0, 359.0)):
        err = abs((got - want + 180) % 360 - 180)
        assert err <= step / 2 + 1e-9
    # Dead-reckoning bytes are untouched.
    assert memory[: layout.num_bytes] == bytearray(layout.num_bytes)


def test_no_heading_store_below_8_bytes():
    assert nav.heading_store_for(7, 6) is None


def simulate(goal, vels, max_days=400):
    """Follow `steer` day by day; return (days taken, final distance)."""
    x, y = 0.0, 0.0
    for day in range(max_days):
        rel = (nav._wrap(goal[0] - x), nav._wrap(goal[1] - y))
        if hypot(*rel) < 25:
            return day, hypot(*rel)
        depth = nav.steer(rel, vels)
        vx, vy = vels[depth]
        x, y = x + vx, y + vy
    return max_days, hypot(*rel)


@pytest.mark.parametrize("vels", [SPREAD, NORTHISH], ids=["spread", "northish"])
def test_steer_reaches_random_goals(vels):
    rng = random.Random(0)
    for _ in range(50):
        goal = (rng.uniform(-500, 500), rng.uniform(-500, 500))
        days, dist = simulate(goal, vels)
        assert dist < 25, f"stuck {dist:.0f} km from {goal}"
        # Never much slower than the planner's own estimate.
        _, est = nav.plan(goal, vels)
        assert days <= est + 5


def test_plan_uses_torus_wrap():
    # Straight south is unreachable with only north-going currents, unless
    # we go north around the torus.
    route = nav.plan((0.0, 100.0), NORTHISH)
    assert route is not None


def test_hold_stays_near_point():
    x, y, worst = 0.0, 0.0, 0.0
    for _ in range(180):
        depth = nav.hold((nav._wrap(-x), nav._wrap(-y)), SPREAD)
        vx, vy = SPREAD[depth]
        x, y = nav._wrap(x + vx), nav._wrap(y + vy)
        worst = max(worst, hypot(x, y))
    assert worst < 45  # within one surface day of the point, never drifting


def test_hold_drifts_slowest_when_it_cannot_stay():
    # With every current heading north, standing still is impossible; the
    # best a jellyfish can do is drift with the slowest current.
    assert nav.hold((0.0, 0.0), NORTHISH) == Depth.DEEP


def test_sweep_alternates_lap_and_side_step():
    depths = [nav.sweep(t, SPREAD) for t in range(200)]
    assert depths[0] == Depth.SURFACE
    assert set(depths) == {Depth.SURFACE, depths[25]}
    assert depths[25] != Depth.SURFACE


@pytest.mark.parametrize("seed", [1, 3, 7])
def test_population_grows_with_8_bytes(seed):
    args = ["--players", "10", "--seed", str(seed), "-T", str(4 * 360)]
    engine = build_engine(parse_args(args))
    engine.run()
    stats = engine.groups[0].stats
    assert stats.extinct_day is None
    assert stats.milestones[10] is not None  # reached 10x the starting size
