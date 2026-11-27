"""Current headings, route planning and holding position for Group 10.

Headings
    A jellyfish only senses the heading of the current it is riding. The four
    headings never change during a run, so an original spends its first days
    visiting each depth once and writes the headings to memory; offspring
    inherit them. Speeds are fixed by the rules, so a heading is all we need.

    Stored right after the dead-reckoning fields:
        6..15 B memory -> bytes 6..7, 4 bits per depth (15 sectors, 24 deg)
        16+ B memory   -> bytes 11..14, 8 bits per depth (255 sectors)
    Value 0 means "not seen yet", so zeroed memory needs no special case.
    Below 8 bytes there is no room, and navigation is disabled.

Route planning
    The jellyfish can only drift, so a trip is a mix of currents. For a goal
    vector g we look for days a, b >= 0 on two currents i, j with
    a*v_i + b*v_j = g and the smallest a + b. Every reachable goal has an
    optimal plan of this two-current form, and on the torus the same goal
    also sits at g + (1000*kx, 1000*ky): we try images up to three laps away,
    nearest first, until no farther image can beat the best route. We re-plan
    every day and ride whichever of the two legs has more days left (if it
    still has a full day), which keeps the path close to the straight line.

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
by scoring tomorrow's position. This module is our own implementation.
"""

from __future__ import annotations

from math import cos, hypot, radians, sin

from core.constants import OCEAN_SIZE
from core.depth import Depth
from core.snapshot import JellyfishSnapshot

Point = tuple[float, float]
DEPTHS = tuple(Depth)

# A plan shorter than this is "basically there": switch to the one-day greedy
# step so we do not overshoot by riding a fast current for a fraction of a day.
_GREEDY_DAYS = 1.5


# ----------------------------------------------------------------- headings
class HeadingStore:
    """Packs the four current headings into memory."""

    def __init__(self, offset: int, bits: int):
        self.offset = offset
        self.bits = bits
        self.sectors = (1 << bits) - 1  # 0 is reserved for "unknown"

    @property
    def end(self) -> int:
        return self.offset + (4 * self.bits + 7) // 8

    def _raw(self, memory: bytearray) -> int:
        return int.from_bytes(memory[self.offset : self.end], "little")

    def read(self, memory: bytearray) -> list[float | None]:
        raw = self._raw(memory)
        mask = (1 << self.bits) - 1
        out: list[float | None] = []
        for d in DEPTHS:
            k = (raw >> (d * self.bits)) & mask
            out.append(None if k == 0 else (k - 1) * 360 / self.sectors)
        return out

    def write(self, memory: bytearray, depth: Depth, heading: float) -> None:
        k = 1 + round(heading * self.sectors / 360) % self.sectors
        shift = depth * self.bits
        mask = ((1 << self.bits) - 1) << shift
        raw = (self._raw(memory) & ~mask) | (k << shift)
        memory[self.offset : self.end] = raw.to_bytes(self.end - self.offset, "little")


def heading_store_for(memory_size: int, dr_bytes: int) -> HeadingStore | None:
    if memory_size >= 16:
        store = HeadingStore(offset=dr_bytes, bits=8)
    else:
        store = HeadingStore(offset=dr_bytes, bits=4)
    return store if store.end <= memory_size else None


def velocity(depth: Depth, heading: float) -> Point:
    """(dx, dy) km/day for a compass heading; x east, y south."""
    rad = radians(heading)
    return depth.speed * sin(rad), -depth.speed * cos(rad)


def known_velocities(
    s: JellyfishSnapshot, headings: list[float | None]
) -> list[Point | None]:
    """Stored velocities, with the current depth's replaced by the exact one."""
    vels: list[Point | None] = [
        None if h is None else velocity(d, h) for d, h in zip(DEPTHS, headings)
    ]
    vels[s.depth] = s.velocity
    return vels


def learn(s: JellyfishSnapshot, store: HeadingStore) -> Depth | None:
    """Record today's heading; return a depth still to visit, or None if all known."""
    headings = store.read(s.memory)
    if headings[s.depth] is None:
        store.write(s.memory, s.depth, s.heading)
        headings[s.depth] = s.heading
    for d in DEPTHS:
        if headings[d] is None:
            return d
    return None


# ------------------------------------------------------------------ planning
def plan(goal: Point, vels: list[Point | None]) -> tuple[Depth | None, float] | None:
    """Today's depth and the total days to reach `goal`, or None if stuck.

    The depth is the leg of the best two-current route with more days left,
    as long as that leg still has a full day. Riding a leg for longer than it
    lasts overshoots sideways, and with a narrow cone of currents that can
    cost a whole extra lap. With less than a day on both legs, the depth is
    None and the caller should finish with greedy one-day steps.
    """
    usable = [(d, v) for d, v in zip(DEPTHS, vels) if v is not None]
    best = _best_route(goal, usable)
    if best is None:
        return None
    total, legs = best
    legs.sort(key=lambda leg: leg[1], reverse=True)
    depth = legs[0][0] if legs[0][1] >= 1 else None
    return depth, total


# Torus images g + 1000 * (kx, ky) to try, nearest first. Seven by seven is
# enough even when every current points into a narrow cone and the goal is
# only reachable after a few laps. Images that cannot beat the best route so
# far are skipped, so in a typical ocean only a handful are ever solved.
_REACH = 3
_IMAGES = sorted(
    (hypot(kx, ky) * OCEAN_SIZE, kx * OCEAN_SIZE, ky * OCEAN_SIZE)
    for kx in range(-_REACH, _REACH + 1)
    for ky in range(-_REACH, _REACH + 1)
)
_MAX_SPEED = Depth.SURFACE.speed
_HALF_DIAGONAL = OCEAN_SIZE / 2 * 2**0.5


def _best_route(
    goal: Point, usable: list[tuple[Depth, Point]]
) -> tuple[float, list[tuple[Depth, float]]] | None:
    pairs = []
    for n, (di, vi) in enumerate(usable):
        for dj, vj in usable[n + 1 :]:
            det = vi[0] * vj[1] - vi[1] * vj[0]
            if abs(det) >= 1e-9:
                pairs.append((di, vi, dj, vj, det))

    best: tuple[float, list[tuple[Depth, float]]] | None = None
    for shift, ox, oy in _IMAGES:
        # |goal| <= half diagonal, so this image is at least this many days away.
        if best is not None and (shift - _HALF_DIAGONAL) / _MAX_SPEED >= best[0]:
            break
        gx, gy = goal[0] + ox, goal[1] + oy
        for di, vi, dj, vj, det in pairs:
            a = (gx * vj[1] - gy * vj[0]) / det
            b = (vi[0] * gy - vi[1] * gx) / det
            if a < -1e-9 or b < -1e-9:
                continue
            if best is None or a + b < best[0]:
                best = (a + b, [(di, a), (dj, b)])
    return best


def hold(goal: Point, vels: list[Point | None]) -> Depth:
    """Depth whose one-day drift ends closest to `goal`."""
    best_depth, best_dist = Depth.DEEP, float("inf")
    for d, v in zip(DEPTHS, vels):
        if v is None:
            continue
        dist = hypot(_wrap(goal[0] - v[0]), _wrap(goal[1] - v[1]))
        if dist < best_dist:
            best_depth, best_dist = d, dist
    return best_depth


def steer(goal: Point, vels: list[Point | None]) -> Depth | None:
    """Travel toward `goal`, finishing with greedy one-day steps."""
    route = plan(goal, vels)
    if route is None:
        return None
    depth, days = route
    if depth is None or days <= _GREEDY_DAYS:
        return hold(goal, vels)
    return depth


# ------------------------------------------------------------------- search
# Lawnmower sweep: one lap of the ocean on the fastest current, then a side
# step on the most perpendicular current, wide enough that the next lap's
# 400 km swath (one zone diameter) overlaps the last one a little.
_LAP_DAYS = round(OCEAN_SIZE / Depth.SURFACE.speed)  # 25
_SIDE_STEP_KM = 350.0
_MAX_SIDE_DAYS = 40


def sweep(day_index: int, vels: list[Point | None]) -> Depth:
    """Depth for an exhaustive search, phased by the calendar (no memory)."""
    main = vels[Depth.SURFACE]
    if main is None:
        return Depth.SURFACE
    ux, uy = main[0] / Depth.SURFACE.speed, main[1] / Depth.SURFACE.speed
    side, side_speed = Depth.SURFACE, 0.0
    for d in DEPTHS[1:]:
        v = vels[d]
        if v is not None and abs(v[0] * uy - v[1] * ux) > side_speed:
            side, side_speed = d, abs(v[0] * uy - v[1] * ux)
    if side_speed < 1.0:
        return Depth.SURFACE  # every current is parallel: nothing better to do
    side_days = min(_MAX_SIDE_DAYS, round(_SIDE_STEP_KM / side_speed))
    phase = day_index % (_LAP_DAYS + side_days)
    return Depth.SURFACE if phase < _LAP_DAYS else side


def _wrap(d: float) -> float:
    d %= OCEAN_SIZE
    return d - OCEAN_SIZE if d > OCEAN_SIZE / 2 else d
