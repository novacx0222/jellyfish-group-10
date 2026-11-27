"""Route planning, holding position and searching for Group 10.

Route planning
    The jellyfish can only drift, so a trip is a mix of currents. For a goal
    vector g we look for days a, b >= 0 on two currents i, j with
    a*v_i + b*v_j = g and the smallest a + b. Every reachable goal has an
    optimal plan of this two-current form, and on the torus the same goal
    also sits at g + (1000*kx, 1000*ky): we try images up to three laps away,
    nearest first, until no farther image can beat the best route. We re-plan
    every day. When interleaving two legs, a low-discrepancy daily phase picks
    the leg, which keeps the path within about a day of the straight line.

Avoiding the predator zone
    A two-current route can be ridden three ways: interleaved (close to the
    straight line), all of leg i first, or all of leg j first. The last two
    are the sides of a parallelogram around the straight line. Once the
    predator centre is known, each candidate costs
        days + PREDATOR_DAY_COST * days spent inside the predator zone
    and we ride the cheapest. A predator day kills 5% of travellers, so a
    long detour is worth it. If the fastest route never enters the zone, it
    is taken as is, so the extra search only runs when it matters.

Holding
    Pick the current that leaves us closest to the target point tomorrow.
    Alternating currents this way keeps the jellyfish within a few km of the
    point instead of drifting off at 5 km/day.

Searching
    A zone never seen is searched for with a lawnmower sweep: a lap on the
    surface current, then a side step on the most perpendicular current. The
    phase comes from the calendar, so it costs no memory.

Ideas also used by other groups: Group 3's notes describe the 3x3 torus images,
the two-current decomposition and a comb-shaped search; Group 5's player holds
by scoring tomorrow's position and detours around danger. This module is our
own implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, floor, hypot, radians, sin, sqrt

from core.constants import OCEAN_SIZE, ZONE_RADIUS
from core.depth import Depth

Point = tuple[float, float]
DEPTHS = tuple(Depth)

# A plan shorter than this is "basically there": switch to the one-day greedy
# step so we do not overshoot by riding a fast current for a fraction of a day.
_GREEDY_DAYS = 1.5

# How many days of extra travel one day inside the predator zone is worth.
PREDATOR_DAY_COST = 6.0
# Our estimate of the predator centre is rough, so keep a little extra berth.
_PREDATOR_BERTH_KM = ZONE_RADIUS + 20.0


def velocity(depth: Depth, heading: float) -> Point:
    """(dx, dy) km/day for a compass heading; x east, y south."""
    rad = radians(heading)
    return depth.speed * sin(rad), -depth.speed * cos(rad)


# ------------------------------------------------------------------ planning
_PHI = 0.6180339887498949
# Days of travel a route "saves" by keeping yesterday's current. Without it,
# two near-equal routes with different first moves can take turns winning
# and the jellyfish dithers in place.
_KEEP_DEPTH_BONUS = 2.0


def phase(day_index: int) -> float:
    """Today's interleaving phase in [0, 1): a golden-ratio (Weyl) sequence."""
    return (day_index * _PHI) % 1.0


@dataclass(frozen=True)
class Route:
    days: float  # total travel days
    legs: tuple[tuple[Depth, float], tuple[Depth, float]]  # (depth, days) x2
    order: str  # "mix", "first", "second": which leg to ride first

    def today(self, day_phase: float = 0.0) -> Depth | None:
        """Depth to ride today, or None if no leg has a full day left.

        "mix" rides leg 1 on days when the phase falls below its share of the
        trip. A low-discrepancy phase spreads those days evenly, so the path
        stays within about a day's drift of the straight line to the goal,
        which is the path `_exposure` scores.
        """
        (d1, a), (d2, b) = self.legs
        if self.order == "first":
            seq = [(d1, a), (d2, b)]
        elif self.order == "second":
            seq = [(d2, b), (d1, a)]
        elif day_phase < a / (a + b):
            seq = [(d1, a), (d2, b)]
        else:
            seq = [(d2, b), (d1, a)]
        for depth, days in seq:
            if days >= 1:
                return depth
        return None


def plan(
    goal: Point,
    vels: list[Point],
    predator: Point | None = None,
    day_phase: float = 0.0,
    current: Depth | None = None,
) -> Route | None:
    """Cheapest route to `goal`, or None if no mix of currents reaches it.

    `predator` is the vector from us to the predator centre, if known, and
    `current` the depth we rode yesterday (keeping it earns a small bonus).
    """
    pairs = _pairs(vels)
    fastest = _fastest(goal, pairs)
    if fastest is None:
        return None
    if predator is None:
        return fastest

    def cost(route: Route) -> float:
        c = route.days + PREDATOR_DAY_COST * _exposure(route, vels, predator)
        if current is not None and route.today(day_phase) == current:
            c -= _KEEP_DEPTH_BONUS
        return c

    best, best_cost, clear = fastest, float("inf"), False
    for route in _orders(fastest):
        c = cost(route)
        if c < best_cost:
            best, best_cost = route, c
        clear = clear or _exposure(route, vels, predator) == 0
    if clear:
        return best  # some way of riding the fastest route stays clear

    # Some detour may be cheaper. A route can only win if its travel time
    # (minus the keep-depth bonus) is below the best cost so far.
    limit = best_cost + _KEEP_DEPTH_BONUS
    candidates = sorted(_feasible(goal, pairs, limit), key=lambda r: r.days)
    for base in candidates:
        if base.days - _KEEP_DEPTH_BONUS >= best_cost:
            break
        for route in _orders(base):
            c = cost(route)
            if c < best_cost:
                best, best_cost = route, c
    return best


# Torus images g + 1000 * (kx, ky) to try, nearest first.
_REACH = 3
_IMAGES = sorted(
    (hypot(kx, ky) * OCEAN_SIZE, kx * OCEAN_SIZE, ky * OCEAN_SIZE)
    for kx in range(-_REACH, _REACH + 1)
    for ky in range(-_REACH, _REACH + 1)
)
_MAX_SPEED = Depth.SURFACE.speed
_HALF_DIAGONAL = OCEAN_SIZE / 2 * 2**0.5


def _pairs(vels: list[Point]):
    out = []
    for n, di in enumerate(DEPTHS):
        vi = vels[di]
        for dj in DEPTHS[n + 1 :]:
            vj = vels[dj]
            det = vi[0] * vj[1] - vi[1] * vj[0]
            if abs(det) >= 1e-9:
                out.append((di, vi, dj, vj, det))
    return out


def _solve(gx: float, gy: float, pair) -> Route | None:
    di, vi, dj, vj, det = pair
    a = (gx * vj[1] - gy * vj[0]) / det
    b = (vi[0] * gy - vi[1] * gx) / det
    if a < -1e-9 or b < -1e-9:
        return None
    return Route(
        max(a, 0.0) + max(b, 0.0), ((di, max(a, 0.0)), (dj, max(b, 0.0))), "mix"
    )


def _fastest(goal: Point, pairs) -> Route | None:
    best: Route | None = None
    for shift, ox, oy in _IMAGES:
        # |goal| <= half diagonal, so this image is at least this many days away.
        if best is not None and (shift - _HALF_DIAGONAL) / _MAX_SPEED >= best.days:
            break
        for pair in pairs:
            route = _solve(goal[0] + ox, goal[1] + oy, pair)
            if route is not None and (best is None or route.days < best.days):
                best = route
    return best


def _feasible(goal: Point, pairs, max_days: float) -> list[Route]:
    out = []
    for shift, ox, oy in _IMAGES:
        if (shift - _HALF_DIAGONAL) / _MAX_SPEED >= max_days:
            break
        for pair in pairs:
            route = _solve(goal[0] + ox, goal[1] + oy, pair)
            if route is not None and route.days < max_days:
                out.append(route)
    return out


def _orders(route: Route) -> list[Route]:
    (_, a), (_, b) = route.legs
    if a < 1 or b < 1:
        return [route]  # one leg is (almost) empty: all orders are the same path
    return [
        route,
        Route(route.days, route.legs, "first"),
        Route(route.days, route.legs, "second"),
    ]


def _exposure(route: Route, vels: list[Point], predator: Point) -> float:
    """Days this route spends inside the predator zone (with some berth)."""
    (d1, a), (d2, b) = route.legs
    v1, v2 = vels[d1], vels[d2]
    end = (v1[0] * a + v2[0] * b, v1[1] * a + v2[1] * b)
    if route.order == "mix":
        return _segment_exposure((0.0, 0.0), end, route.days, predator)
    if route.order == "first":
        mid, first_days, second_days = (v1[0] * a, v1[1] * a), a, b
    else:
        mid, first_days, second_days = (v2[0] * b, v2[1] * b), b, a
    return _segment_exposure((0.0, 0.0), mid, first_days, predator) + _segment_exposure(
        mid, end, second_days, predator
    )


def _segment_exposure(p: Point, q: Point, days: float, c: Point) -> float:
    """Days spent within the predator berth along p -> q, all torus images."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0 or days <= 0:
        return 0.0
    r = _PREDATOR_BERTH_KM
    total = 0.0
    kx_lo = ceil((min(p[0], q[0]) - r - c[0]) / OCEAN_SIZE)
    kx_hi = floor((max(p[0], q[0]) + r - c[0]) / OCEAN_SIZE)
    ky_lo = ceil((min(p[1], q[1]) - r - c[1]) / OCEAN_SIZE)
    ky_hi = floor((max(p[1], q[1]) + r - c[1]) / OCEAN_SIZE)
    for kx in range(kx_lo, kx_hi + 1):
        for ky in range(ky_lo, ky_hi + 1):
            fx = p[0] - (c[0] + kx * OCEAN_SIZE)
            fy = p[1] - (c[1] + ky * OCEAN_SIZE)
            # |f + t d|^2 = r^2  ->  t^2 |d|^2 + 2 t (f.d) + |f|^2 - r^2 = 0
            half_b = fx * dx + fy * dy
            disc = half_b * half_b - length_sq * (fx * fx + fy * fy - r * r)
            if disc <= 0:
                continue
            root = sqrt(disc)
            lo = max(0.0, (-half_b - root) / length_sq)
            hi = min(1.0, (-half_b + root) / length_sq)
            if hi > lo:
                total += (hi - lo) * days
    return total


# ------------------------------------------------------------------- moving
def hold(goal: Point, vels: list[Point]) -> Depth:
    """Depth whose one-day drift ends closest to `goal`."""
    best_depth, best_dist = Depth.DEEP, float("inf")
    for d, v in zip(DEPTHS, vels):
        dist = hypot(_wrap(goal[0] - v[0]), _wrap(goal[1] - v[1]))
        if dist < best_dist:
            best_depth, best_dist = d, dist
    return best_depth


def steer(
    goal: Point,
    vels: list[Point],
    predator: Point | None = None,
    day_phase: float = 0.0,
    current: Depth | None = None,
) -> Depth | None:
    """Travel toward `goal`, finishing with greedy one-day steps."""
    route = plan(goal, vels, predator, day_phase, current)
    if route is None:
        return None
    depth = route.today(day_phase)
    if depth is None or route.days <= _GREEDY_DAYS:
        return hold(goal, vels)
    return depth


# ------------------------------------------------------------------- search
# Lawnmower sweep: one lap of the ocean on the fastest current, then a side
# step on the most perpendicular current, wide enough that the next lap's
# 400 km swath (one zone diameter) overlaps the last one a little.
_LAP_DAYS = round(OCEAN_SIZE / Depth.SURFACE.speed)  # 25
_SIDE_STEP_KM = 350.0
_MAX_SIDE_DAYS = 40


def sweep(day_index: int, vels: list[Point]) -> Depth:
    """Depth for an exhaustive search, phased by the calendar (no memory)."""
    main = vels[Depth.SURFACE]
    ux, uy = main[0] / Depth.SURFACE.speed, main[1] / Depth.SURFACE.speed
    side, side_speed = Depth.SURFACE, 0.0
    for d in DEPTHS[1:]:
        v = vels[d]
        if abs(v[0] * uy - v[1] * ux) > side_speed:
            side, side_speed = d, abs(v[0] * uy - v[1] * ux)
    if side_speed < 1.0:
        return Depth.SURFACE  # every current is parallel: nothing better to do
    side_days = min(_MAX_SIDE_DAYS, round(_SIDE_STEP_KM / side_speed))
    phase = day_index % (_LAP_DAYS + side_days)
    return Depth.SURFACE if phase < _LAP_DAYS else side


def _wrap(d: float) -> float:
    d %= OCEAN_SIZE
    return d - OCEAN_SIZE if d > OCEAN_SIZE / 2 else d
