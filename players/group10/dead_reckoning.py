"""Dead reckoning: where am I, and where are the feeding / breeding zones?

A jellyfish cannot sense its (x, y) position, but `snapshot.velocity` is the
exact drift it made yesterday. Summing those drifts gives its position in a
private "lineage frame" whose origin is where the original jellyfish was born.
Offspring are born at the parent's position with a copy of its memory, so a
whole lineage shares one frame, and zone estimates found by an ancestor stay
valid for every descendant.

Memory layout (bit-packed, little-endian, starting at byte 0):

    pos_x, pos_y          POS_BITS each   our position, in cells
    feed_known            1 bit
    feed_x, feed_y        MARK_BITS each  feeding-zone centre estimate
    breed_known           1 bit
    breed_x, breed_y      MARK_BITS each  breeding-zone centre estimate

The ocean is 1000 km and every field spans exactly one lap of it, so torus
wrap-around is just integer overflow. Zeroed memory decodes as "at the origin,
no zones known", which is exactly the state of a newly born original.

Two layouts, picked by memory size:

    memory  POS_BITS  MARK_BITS  bytes  position cell  zone cell
    6..15       9         7        6      1.95 km       7.8 km
    16+        16        12       11      0.015 km      0.24 km

With 8 bytes, bytes 6..7 stay free for later features. Below 6 bytes dead
reckoning is disabled.

Rounding: a current's per-day drift is a fixed fraction of a cell, so plain
rounding would add the same error every day and the estimate would walk away.
Instead each day's drift is rounded with a deterministic dither that follows a
golden-ratio (Weyl) sequence over the day index. The rounding errors then
average out over time, and no `self.rng` draws are needed.

Zone-centre estimates:
- First sighting from outside: the centre lies somewhere ahead of us. Its
  mean offset along the entry direction is R * 2/pi, so we start there. An
  original jellyfish that starts inside the breeding zone uses its birth point.
- Every later day: if "inside the zone?" contradicts the estimate (inside
  but the estimate is more than R away, or outside but less than R), move
  the estimate along the line to us until it is consistent again. The idea
  of correcting a centre from in/out readings comes from Group 5's player;
  this frame, layout, dither and update rule are our own implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, hypot, pi

from core.constants import OCEAN_SIZE, ZONE_RADIUS
from core.snapshot import JellyfishSnapshot

# Golden-ratio conjugate: frac(t * PHI) is a low-discrepancy sequence in [0, 1).
_PHI = 0.6180339887498949
# Y uses a shifted copy of the sequence so the two axes do not round together.
_Y_DITHER_OFFSET = 0.5

# How far past the boundary a corrected centre is placed. A few cells of
# slack keeps one reading from flipping the estimate back and forth.
_MARGIN_KM = 10.0
# Mean distance, along the entry direction, from an entry point to the centre.
_ENTRY_INSET_KM = ZONE_RADIUS * 2 / pi

FEEDING = "feeding"
BREEDING = "breeding"
_ZONES = (FEEDING, BREEDING)

Point = tuple[float, float]


def _wrap_delta(d: float) -> float:
    """Shortest signed displacement along one axis of the torus."""
    d %= OCEAN_SIZE
    return d - OCEAN_SIZE if d > OCEAN_SIZE / 2 else d


@dataclass(frozen=True)
class Layout:
    pos_bits: int
    mark_bits: int

    @property
    def total_bits(self) -> int:
        return 2 * self.pos_bits + len(_ZONES) * (1 + 2 * self.mark_bits)

    @property
    def num_bytes(self) -> int:
        return (self.total_bits + 7) // 8

    @property
    def pos_cell_km(self) -> float:
        return OCEAN_SIZE / (1 << self.pos_bits)

    @property
    def mark_cell_km(self) -> float:
        return OCEAN_SIZE / (1 << self.mark_bits)


SMALL = Layout(pos_bits=9, mark_bits=7)  # 6 bytes
LARGE = Layout(pos_bits=16, mark_bits=12)  # 11 bytes


def layout_for(memory_size: int) -> Layout | None:
    if memory_size >= 16:
        return LARGE
    if memory_size >= SMALL.num_bytes:
        return SMALL
    return None


@dataclass
class Chart:
    """Decoded dead-reckoning state. Positions are km in the lineage frame."""

    pos_x: int  # cells
    pos_y: int  # cells
    zones: dict[str, Point | None]

    def position(self, layout: Layout) -> Point:
        return self.pos_x * layout.pos_cell_km, self.pos_y * layout.pos_cell_km

    def vector_to(self, zone: str, layout: Layout) -> Point | None:
        """Shortest (dx, dy) in km from us to a zone centre, or None if unknown."""
        centre = self.zones[zone]
        if centre is None:
            return None
        x, y = self.position(layout)
        return _wrap_delta(centre[0] - x), _wrap_delta(centre[1] - y)


# --------------------------------------------------------------- bit packing
def decode(memory: bytearray, layout: Layout) -> Chart:
    bits = int.from_bytes(memory[: layout.num_bytes], "little")

    def take(n: int) -> int:
        nonlocal bits
        value = bits & ((1 << n) - 1)
        bits >>= n
        return value

    pos_x = take(layout.pos_bits)
    pos_y = take(layout.pos_bits)
    zones: dict[str, Point | None] = {}
    for zone in _ZONES:
        known = take(1)
        x = take(layout.mark_bits) * layout.mark_cell_km
        y = take(layout.mark_bits) * layout.mark_cell_km
        zones[zone] = (x, y) if known else None
    return Chart(pos_x, pos_y, zones)


def encode(chart: Chart, memory: bytearray, layout: Layout) -> None:
    bits = 0
    shift = 0

    def put(value: int, n: int) -> None:
        nonlocal bits, shift
        bits |= (value % (1 << n)) << shift
        shift += n

    put(chart.pos_x, layout.pos_bits)
    put(chart.pos_y, layout.pos_bits)
    cells = 1 << layout.mark_bits
    for zone in _ZONES:
        centre = chart.zones[zone]
        put(centre is not None, 1)
        x, y = centre or (0.0, 0.0)
        put(round(x / layout.mark_cell_km) % cells, layout.mark_bits)
        put(round(y / layout.mark_cell_km) % cells, layout.mark_bits)

    # Keep any bits above our fields (the last byte may be shared later).
    n = layout.num_bytes
    high = int.from_bytes(memory[:n], "little") >> shift << shift
    memory[:n] = (high | bits).to_bytes(n, "little")


# ------------------------------------------------------------------- update
def is_first_turn(s: JellyfishSnapshot) -> bool:
    """True if the jellyfish has not drifted since birth.

    Originals act first on their birth day (age 0). A newborn acts first the
    day after birth (age 1), but it skipped the parent's drift on its birth
    day, so `velocity` does not describe a move it made. Only originals are
    born on day 1 of year 1 (births happen in winter).
    """
    if s.age_days == 0:
        return True
    is_original = (s.birth_year, s.birth_day_of_year) == (1, 1)
    return s.age_days == 1 and not is_original


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
        target = ZONE_RADIUS - _MARGIN_KM
    elif not inside and dist < ZONE_RADIUS:
        target = ZONE_RADIUS + _MARGIN_KM
        if dist == 0:
            # Sitting on the old estimate yet outside: push it back along
            # the way we came, since we just left through that side.
            vx, vy = s.velocity
            speed = hypot(vx, vy)
            if speed == 0:
                return
            dx, dy, dist = -vx, -vy, speed
    else:
        return
    k = target / dist
    chart.zones[zone] = ((x + dx * k) % OCEAN_SIZE, (y + dy * k) % OCEAN_SIZE)


def update(s: JellyfishSnapshot, layout: Layout) -> Chart:
    """Apply yesterday's drift and today's zone readings to memory.

    Call exactly once per jellyfish per day, at the start of `act`.
    """
    chart = decode(s.memory, layout)
    first_turn = is_first_turn(s)
    if not first_turn:
        _advance(chart, s, layout)
    _observe(chart, FEEDING, s.in_feeding_zone, s, first_turn, layout)
    _observe(chart, BREEDING, s.in_breeding_zone, s, first_turn, layout)
    encode(chart, s.memory, layout)
    return chart
