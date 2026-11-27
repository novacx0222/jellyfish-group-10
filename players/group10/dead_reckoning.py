"""The jellyfish's chart: where am I, where are the zones, how do currents flow?

A jellyfish cannot sense its (x, y) position, but `snapshot.velocity` is the
exact drift it made yesterday. Summing those drifts gives its position in a
private "lineage frame" whose origin is where the original jellyfish was born.
Offspring are born at the parent's position with a copy of its memory, so a
whole lineage shares one frame, and zone estimates found by an ancestor stay
valid for every descendant.

Memory layout: one bit-packed little-endian integer starting at byte 0.

    pos_x, pos_y                 POS bits each    our position, in cells
    feed_known, feed_x, feed_y   1 + MARK + MARK  feeding-zone point
    breed_x, breed_y             MARK + MARK      breeding-zone point
    pred_known, pred_x, pred_y   1 + PRED + PRED  predator-zone centre
    heading[4]                   HEAD bits each   current headings by depth

Every coordinate field spans exactly one lap of the 1000 km ocean, so torus
wrap-around is just integer overflow. Zeroed memory decodes as "at the origin,
breeding zone at the origin, nothing else known", which is exactly the state
of a newly born original: it starts inside the breeding zone, so the breeding
zone needs no "known" flag.

Headings need no "known" flag either: an original learns them on a fixed
schedule (depth d on day d of its life, see `headings_known`), and offspring
inherit all four.

    memory   POS  MARK  PRED  HEAD   bits  position cell  zone cell  pred cell
    8..11     8    5     5     4      64      3.9 km       31 km      31 km
    12..16   10    8     7     6      92      1.0 km       3.9 km     7.8 km
    17+      16   12    10     8     134      0.015 km     0.24 km    1.0 km

Below 8 bytes the chart is disabled.

Rounding: a current's per-day drift is a fixed fraction of a cell, so plain
rounding would add the same error every day and the estimate would walk away.
Instead each day's drift is rounded with a deterministic dither that follows a
golden-ratio (Weyl) sequence over the day index. The rounding errors then
average out over time, and no `self.rng` draws are needed.

Zone estimates:
- First sighting from outside: the centre lies somewhere ahead of us. Its
  mean offset along the entry direction is R * 2/pi, so we start there.
- Every later day: if "inside the zone?" contradicts the estimate (inside
  but the estimate is more than R away, or outside but less than R), move
  the estimate along the line to us until it is consistent again. If we are
  outside but almost on top of the estimate, that line is noise; we most
  likely just stepped out, so the estimate goes back along our last drift.
The feeding and breeding points double as hold points: the player may move
them off a predator overlap, and these corrections keep them inside the zone.

The idea of correcting a centre from in/out readings comes from Group 5's
player; this frame, layout, dither and update rule are our own implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, floor, hypot, pi, sin, tau

from core.constants import CURRENT_SPEEDS, OCEAN_SIZE, ZONE_RADIUS
from core.depth import Depth
from core.snapshot import JellyfishSnapshot

# Golden-ratio conjugate: frac(t * PHI) is a low-discrepancy sequence in [0, 1).
_PHI = 0.6180339887498949
# Y uses a shifted copy of the sequence so the two axes do not round together.
_Y_DITHER_OFFSET = 0.5

# Mean distance, along the entry direction, from an entry point to the centre.
_ENTRY_INSET_KM = ZONE_RADIUS * 2 / pi

FEEDING = "feeding"
BREEDING = "breeding"
PREDATOR = "predator"
_ZONES = (FEEDING, BREEDING, PREDATOR)

Point = tuple[float, float]


def _wrap_delta(d: float) -> float:
    """Shortest signed displacement along one axis of the torus."""
    d %= OCEAN_SIZE
    return d - OCEAN_SIZE if d > OCEAN_SIZE / 2 else d


@dataclass(frozen=True)
class Layout:
    pos_bits: int
    mark_bits: int  # feeding and breeding points
    pred_bits: int  # predator centre
    heading_bits: int

    def __post_init__(self) -> None:
        # Everything below is derived from the four widths. It is computed
        # once here because `act` reads it millions of times per run.
        specs = tuple(
            (
                z,
                self.has_flag(z),
                self.zone_bits(z),
                OCEAN_SIZE / (1 << self.zone_bits(z)),
            )
            for z in _ZONES
        )
        total = 2 * self.pos_bits + 4 * self.heading_bits
        total += sum(flag + 2 * n for _, flag, n, _ in specs)
        sectors = 1 << self.heading_bits
        velocities = tuple(
            tuple(
                (speed * sin(tau * k / sectors), -speed * cos(tau * k / sectors))
                for k in range(sectors)
            )
            for speed in CURRENT_SPEEDS
        )
        derived = {
            "zone_specs": specs,
            "total_bits": total,
            "num_bytes": (total + 7) // 8,
            "pos_cell_km": OCEAN_SIZE / (1 << self.pos_bits),
            "cell_km": {z: cell for z, _, _, cell in specs},
            "sectors": sectors,
            "velocities": velocities,  # [depth][sector] -> (dx, dy) km/day
        }
        for name, value in derived.items():
            object.__setattr__(self, name, value)

    def zone_bits(self, zone: str) -> int:
        return self.pred_bits if zone == PREDATOR else self.mark_bits

    @staticmethod
    def has_flag(zone: str) -> bool:
        return zone != BREEDING

    def zone_cell_km(self, zone: str) -> float:
        return self.cell_km[zone]


SMALL = Layout(pos_bits=8, mark_bits=5, pred_bits=5, heading_bits=4)  # 8 bytes
MEDIUM = Layout(pos_bits=10, mark_bits=8, pred_bits=7, heading_bits=6)  # 12 bytes
LARGE = Layout(pos_bits=16, mark_bits=12, pred_bits=10, heading_bits=8)  # 17 bytes
_LAYOUTS = (LARGE, MEDIUM, SMALL)  # richest first


def layout_for(memory_size: int) -> Layout | None:
    for layout in _LAYOUTS:
        if layout.num_bytes <= memory_size:
            return layout
    return None


@dataclass
class Chart:
    """Decoded memory. Zone points are km in the lineage frame."""

    pos_x: int  # cells
    pos_y: int  # cells
    zones: dict[str, Point | None]
    headings: list[int] = field(default_factory=lambda: [0, 0, 0, 0])  # sectors

    def position(self, layout: Layout) -> Point:
        cell = layout.pos_cell_km
        return self.pos_x * cell, self.pos_y * cell

    def vector_to(self, zone: str, layout: Layout) -> Point | None:
        """Shortest (dx, dy) in km from us to a zone point, or None if unknown."""
        point = self.zones[zone]
        if point is None:
            return None
        cell = layout.pos_cell_km
        return (
            _wrap_delta(point[0] - self.pos_x * cell),
            _wrap_delta(point[1] - self.pos_y * cell),
        )

    def velocities(self, layout: Layout) -> list[Point]:
        """Stored velocity of every current, by depth."""
        table = layout.velocities
        return [table[d][k] for d, k in enumerate(self.headings)]

    def heading(self, depth: Depth, layout: Layout) -> float:
        return self.headings[depth] * 360 / layout.sectors

    def set_heading(self, depth: Depth, heading: float, layout: Layout) -> None:
        self.headings[depth] = round(heading * layout.sectors / 360) % layout.sectors


# --------------------------------------------------------------- bit packing
def decode(memory: bytearray, layout: Layout) -> Chart:
    bits = int.from_bytes(memory[: layout.num_bytes], "little")
    n = layout.pos_bits
    mask = (1 << n) - 1
    pos_x = bits & mask
    pos_y = (bits >> n) & mask
    bits >>= 2 * n
    zones: dict[str, Point | None] = {}
    for zone, flag, n, cell in layout.zone_specs:
        known = True
        if flag:
            known = bits & 1
            bits >>= 1
        mask = (1 << n) - 1
        if known:
            zones[zone] = ((bits & mask) * cell, ((bits >> n) & mask) * cell)
        else:
            zones[zone] = None
        bits >>= 2 * n
    n = layout.heading_bits
    mask = (1 << n) - 1
    headings = [(bits >> (i * n)) & mask for i in range(4)]
    return Chart(pos_x, pos_y, zones, headings)


def encode(chart: Chart, memory: bytearray, layout: Layout) -> None:
    n = layout.pos_bits
    mask = (1 << n) - 1
    bits = (chart.pos_x & mask) | ((chart.pos_y & mask) << n)
    shift = 2 * n
    for zone, flag, n, cell in layout.zone_specs:
        point = chart.zones[zone]
        if flag:
            bits |= (point is not None) << shift
            shift += 1
        if point is not None:
            mask = (1 << n) - 1
            bits |= (round(point[0] / cell) & mask) << shift
            bits |= (round(point[1] / cell) & mask) << (shift + n)
        shift += 2 * n
    n = layout.heading_bits
    mask = (1 << n) - 1
    for sector in chart.headings:
        bits |= (sector & mask) << shift
        shift += n

    # Keep any bits above our fields (the last byte may be shared later).
    size = layout.num_bytes
    high = int.from_bytes(memory[:size], "little") >> shift << shift
    memory[:size] = (high | bits).to_bytes(size, "little")


# ------------------------------------------------------------------- update
def is_original(s: JellyfishSnapshot) -> bool:
    """Only the initial jellyfish are born on day 1 of year 1 (births are in winter)."""
    return (s.birth_year, s.birth_day_of_year) == (1, 1)


def is_first_turn(s: JellyfishSnapshot) -> bool:
    """True if the jellyfish has not drifted since birth.

    Originals act first on their birth day (age 0). A newborn acts first the
    day after birth (age 1), but it skipped the parent's drift on its birth
    day, so `velocity` does not describe a move it made.
    """
    if s.age_days == 0:
        return True
    return s.age_days == 1 and not is_original(s)


def headings_known(s: JellyfishSnapshot) -> int:
    """How many depths (in Depth order) have a stored heading.

    An original starts at the surface and the player sends it one depth deeper
    each day, recording the current it rides: depth d is known from age d + 1.
    """
    if not is_original(s):
        return 4
    return min(4, s.age_days)


def _dither(s: JellyfishSnapshot, offset: float) -> float:
    t = (s.year - 1) * 360 + s.day_of_year
    return (t * _PHI + offset) % 1.0


def _advance(chart: Chart, s: JellyfishSnapshot, layout: Layout) -> None:
    vx, vy = s.velocity
    cell = layout.pos_cell_km
    chart.pos_x += floor(vx / cell + _dither(s, 0.0))
    chart.pos_y += floor(vy / cell + _dither(s, _Y_DITHER_OFFSET))
    cells = 1 << layout.pos_bits
    chart.pos_x %= cells
    chart.pos_y %= cells


def _observe(
    chart: Chart,
    zone: str,
    inside: bool,
    s: JellyfishSnapshot,
    first_turn: bool,
    layout: Layout,
) -> None:
    x, y = chart.position(layout)
    vec = chart.vector_to(zone, layout)
    # Corrections land this far past the boundary: a few km, and at least
    # half a stored cell so rounding cannot undo them.
    margin = max(10.0, 0.6 * layout.zone_cell_km(zone))

    if vec is None:
        if not inside:
            return
        vx, vy = (0.0, 0.0) if first_turn else s.velocity
        speed = hypot(vx, vy)
        if speed == 0:
            chart.zones[zone] = (x, y)
        else:
            k = _ENTRY_INSET_KM / speed
            chart.zones[zone] = ((x + vx * k) % OCEAN_SIZE, (y + vy * k) % OCEAN_SIZE)
        return

    dx, dy = vec
    dist = hypot(dx, dy)
    if inside and dist > ZONE_RADIUS:
        target = ZONE_RADIUS - margin
    elif not inside and dist < ZONE_RADIUS:
        target = ZONE_RADIUS + margin
        if dist < ZONE_RADIUS / 2 and not first_turn:
            # Outside, yet close to the estimate: the direction to it is
            # mostly noise. We most likely just stepped out of the zone, so
            # the centre is behind us: push it back along the way we came.
            vx, vy = s.velocity
            speed = hypot(vx, vy)
            if speed == 0:
                return
            dx, dy, dist = -vx, -vy, speed
    else:
        return
    k = target / dist
    chart.zones[zone] = ((x + dx * k) % OCEAN_SIZE, (y + dy * k) % OCEAN_SIZE)


def update(chart: Chart, s: JellyfishSnapshot, layout: Layout) -> None:
    """Apply yesterday's drift and today's zone readings to a decoded chart.

    Call exactly once per jellyfish per day, at the start of `act`, then
    `encode` the chart back once the player is done with it.
    """
    first_turn = is_first_turn(s)
    if not first_turn:
        _advance(chart, s, layout)
    _observe(chart, FEEDING, s.in_feeding_zone, s, first_turn, layout)
    _observe(chart, BREEDING, s.in_breeding_zone, s, first_turn, layout)
    _observe(chart, PREDATOR, s.in_predator_zone, s, first_turn, layout)
