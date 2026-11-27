"""Tests for Group 10 route planning, predator detours, holding and search.

Run with: uv run python -m pytest players/group10/tests
"""

import random
from math import hypot

import pytest

from core.constants import ZONE_RADIUS
from core.depth import Depth
from core.parse_args import parse_args
from core.runner import build_engine
from players.group10 import navigation as nav

# Four currents spanning the plane.
SPREAD = [nav.velocity(d, h) for d, h in zip(nav.DEPTHS, (45, 135, 225, 315))]
# Every current heads roughly north: most goals need a trip around the torus.
NORTHISH = [nav.velocity(d, h) for d, h in zip(nav.DEPTHS, (344, 341, 20, 31))]


def simulate(goal, vels, predator=None, tell=True, max_days=400):
    """Follow `steer` day by day from the origin.

    Returns (days taken, final distance, days started inside the predator
    zone). `predator` is a fixed point relative to the start; with
    `tell=False` exposure is counted but the planner is not told about it.
    """
    x, y, exposed = 0.0, 0.0, 0
    for day in range(max_days):
        rel = (nav._wrap(goal[0] - x), nav._wrap(goal[1] - y))
        if hypot(*rel) < 25:
            return day, hypot(*rel), exposed
        pred = None
        if predator is not None:
            pred = (nav._wrap(predator[0] - x), nav._wrap(predator[1] - y))
            exposed += hypot(*pred) < ZONE_RADIUS
        depth = nav.steer(rel, vels, pred if tell else None, nav.phase(day))
        vx, vy = vels[depth]
        x, y = x + vx, y + vy
    return max_days, hypot(*rel), exposed


@pytest.mark.parametrize("vels", [SPREAD, NORTHISH], ids=["spread", "northish"])
def test_steer_reaches_random_goals(vels):
    rng = random.Random(0)
    for _ in range(50):
        goal = (rng.uniform(-500, 500), rng.uniform(-500, 500))
        days, dist, _ = simulate(goal, vels)
        assert dist < 25, f"stuck {dist:.0f} km from {goal}"
        # Never much slower than the planner's own estimate.
        assert days <= nav.plan(goal, vels).days + 5


def test_plan_uses_torus_wrap():
    # Straight south is unreachable with only north-going currents, unless
    # we go north around the torus.
    route = nav.plan((0.0, 100.0), NORTHISH)
    assert route is not None


def test_route_detours_around_predator():
    # Goal due east; the predator zone sits right on the straight line.
    goal, predator = (500.0, 0.0), (250.0, 0.0)
    _, _, straight = simulate(goal, SPREAD, predator, tell=False)
    _, dist, exposed = simulate(goal, SPREAD, predator)
    assert dist < 25
    assert straight > 0 and exposed < straight / 2


def test_predator_off_route_changes_nothing():
    goal = (500.0, 0.0)
    far = (0.0, 500.0)
    assert nav.plan(goal, SPREAD, far) == nav.plan(goal, SPREAD)


@pytest.mark.parametrize("vels", [SPREAD, NORTHISH], ids=["spread", "northish"])
def test_steer_with_predator_still_arrives(vels):
    rng = random.Random(2)
    for _ in range(30):
        goal = (rng.uniform(-500, 500), rng.uniform(-500, 500))
        predator = (rng.uniform(-500, 500), rng.uniform(-500, 500))
        _, dist, _ = simulate(goal, vels, predator)
        assert dist < 25


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
