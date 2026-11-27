"""Tests for Group 10 dead reckoning and the memory layout.

Run with: uv run python -m pytest players/group10/tests
"""

import random
from math import hypot
from types import SimpleNamespace

import pytest

from core.constants import ZONE_RADIUS
from core.depth import Depth
from core.engine import Engine, Group, SimConfig
from core.geometry import torus_delta, torus_distance, wrap
from core.ocean import Ocean
from core.runner import spawn_points
from core.stats import GroupStats
from players.group10 import dead_reckoning as dr
from players.group10.player import Player10


# ------------------------------------------------------------- unit tests
@pytest.mark.parametrize("memory_size", [8, 11, 12, 16, 17, 1024])
def test_encode_decode_round_trip(memory_size):
    layout = dr.layout_for(memory_size)
    assert layout is not None and layout.num_bytes <= memory_size
    memory = bytearray(memory_size)
    chart = dr.Chart(
        pos_x=(1 << layout.pos_bits) - 3,
        pos_y=7,
        zones={dr.FEEDING: (123.0, 987.0), dr.BREEDING: (500.0, 250.0)}
        | {dr.PREDATOR: None},
    )
    for depth, heading in zip(Depth, (0.0, 97.0, 200.0, 359.0)):
        chart.set_heading(depth, heading, layout)
    dr.encode(chart, memory, layout)
    back = dr.decode(memory, layout)
    assert (back.pos_x, back.pos_y) == (chart.pos_x, chart.pos_y)
    assert back.zones[dr.PREDATOR] is None
    for zone, (x, y) in ((dr.FEEDING, (123.0, 987.0)), (dr.BREEDING, (500.0, 250.0))):
        bx, by = back.zones[zone]
        half_cell = layout.zone_cell_km(zone) / 2 + 1e-9
        # Compare on the torus: 987 km may round up to 1000 km == 0 km.
        assert abs(torus_delta(x, bx)) <= half_cell
        assert abs(torus_delta(y, by)) <= half_cell
    step = 360 / layout.sectors
    for depth, want in zip(Depth, (0.0, 97.0, 200.0, 359.0)):
        err = abs((back.heading(depth, layout) - want + 180) % 360 - 180)
        assert err <= step / 2 + 1e-9


def test_layouts_fit_their_memory():
    assert dr.SMALL.total_bits <= 64
    assert dr.layout_for(8) is dr.SMALL
    assert dr.layout_for(12) is dr.MEDIUM
    assert dr.layout_for(17) is dr.LARGE


def test_zeroed_memory_is_a_newborn_original():
    chart = dr.decode(bytearray(8), dr.SMALL)
    assert (chart.pos_x, chart.pos_y) == (0, 0)
    # Originals are born in the breeding zone, so it starts "at the origin".
    assert chart.zones[dr.BREEDING] == (0.0, 0.0)
    assert chart.zones[dr.FEEDING] is None and chart.zones[dr.PREDATOR] is None


def test_encode_keeps_bits_after_layout():
    memory = bytearray(b"\x00" * 8 + b"\xab\xcd")
    dr.encode(dr.decode(memory, dr.SMALL), memory, dr.SMALL)
    assert memory[8:] == b"\xab\xcd"
    # MEDIUM uses 92 bits: the top 4 bits of byte 11 belong to someone else.
    memory = bytearray(11) + b"\xf0"
    dr.encode(dr.decode(memory, dr.MEDIUM), memory, dr.MEDIUM)
    assert memory[11] == 0xF0


def test_too_small_memory_disables_the_chart():
    assert dr.layout_for(7) is None
    player = Player10(1, 1080, random.Random(0))
    assert player.layout is None


def test_headings_known_schedule():
    snap = SimpleNamespace(age_days=0, birth_year=1, birth_day_of_year=1)
    assert [dr.headings_known(snap)] == [0]
    snap.age_days = 3
    assert dr.headings_known(snap) == 3
    snap.age_days = 50
    assert dr.headings_known(snap) == 4
    newborn = SimpleNamespace(age_days=1, birth_year=2, birth_day_of_year=300)
    assert dr.headings_known(newborn) == 4  # inherited from the parent


# -------------------------------------------------------- engine harness
def run_and_measure(seed: int, memory_size: int, days: int):
    """Run Player10 in the real engine and compare its chart to the truth.

    Returns, in km, one sample per living jellyfish per day:
      - position error: believed position vs. true position, and
      - feeding / breeding / predator error: the believed vector from the
        jellyfish to the zone point vs. the true vector to the zone centre.
        This is what navigation uses, and in/out corrections re-anchor it
        even while the position drifts.
    """
    rng = random.Random(seed)
    ocean = Ocean.generate(rng)
    starts = spawn_points(rng, ocean, 10)
    config = SimConfig(memory_size, 3 * 360, len(starts), days, 2000)
    player = Player10(memory_size, config.lifetime_days, random.Random(seed))
    group = Group(0, "10", "G10", player, random.Random(seed), GroupStats(10))
    engine = Engine(ocean, [group], config, starts)
    layout = player.layout

    origin: dict[int, tuple[float, float]] = {j.id: (j.x, j.y) for j in group.alive}
    pos_err, feed_err, breed_err, pred_err = [], [], [], []
    while not engine.finished:
        engine.step()
        for j in group.alive:
            if j.id not in origin:
                origin[j.id] = origin[j.parent_id]  # same lineage frame
            ox, oy = origin[j.id]
            chart = dr.decode(j.memory, layout)
            x, y = chart.position(layout)
            # Memory reflects the start of the day, before the drift the
            # engine just applied; newborns have not drifted at all.
            if j.birth_t == engine.t - 1 and j.generation > 0:
                tx, ty = j.x, j.y
            else:
                dx, dy = ocean.currents[j.depth].velocity
                tx, ty = wrap(j.x - dx), wrap(j.y - dy)
            pos_err.append(torus_distance(wrap(ox + x), wrap(oy + y), tx, ty))
            for zone, true_zone, errs in (
                (dr.FEEDING, ocean.feeding, feed_err),
                (dr.BREEDING, ocean.breeding, breed_err),
                (dr.PREDATOR, ocean.predator, pred_err),
            ):
                est = chart.vector_to(zone, layout)
                if est is not None:
                    true_dx = torus_delta(tx, true_zone.x)
                    true_dy = torus_delta(ty, true_zone.y)
                    # Compare modulo the torus: near the antipode the two
                    # vectors may legitimately wrap to opposite sides.
                    err_x = torus_delta(true_dx, est[0])
                    err_y = torus_delta(true_dy, est[1])
                    errs.append(hypot(err_x, err_y))
    return pos_err, feed_err, breed_err, pred_err


@pytest.mark.parametrize("memory_size", [8, 12, 1024])
@pytest.mark.parametrize("seed", [2, 8])
def test_position_tracks_truth(seed, memory_size):
    pos_err, *_ = run_and_measure(seed, memory_size, 3 * 360)
    assert pos_err, "no jellyfish survived to be measured"
    # Cells are 3.9 km at 8 B (8 bits per axis, to make room for the
    # predator zone) and 0.015 km at 1 KB. The dither keeps rounding errors
    # from compounding, but at 8 B the raw position still drifts: up to ~125 km
    # over three years in our runs. Navigation uses zone vectors, which every
    # in/out reading re-anchors (see the next test), so this is tolerable.
    limit = {8: 150.0, 12: 40.0}.get(memory_size, 1.0)
    assert max(pos_err) < limit


@pytest.mark.parametrize("seed", [1, 8])
def test_zone_points_stay_inside_zones(seed):
    _, feed_err, breed_err, pred_err = run_and_measure(seed, 8, 3 * 360)
    assert feed_err and breed_err and pred_err
    # The stored point is where the jellyfish heads and holds. It starts as a
    # centre estimate and may later move off a predator overlap, so what
    # matters is that it lies inside the true zone. Measured: >= 99% here.
    for errs in (feed_err, breed_err, pred_err):
        inside = sum(e < ZONE_RADIUS for e in errs) / len(errs)
        assert inside > 0.95


def test_first_turn_detection():
    """No dead-reckoning step on a jellyfish's first act after birth."""
    snap = SimpleNamespace(age_days=0, birth_year=1, birth_day_of_year=1)
    assert dr.is_first_turn(snap)  # original, day 1
    snap.age_days = 1
    assert not dr.is_first_turn(snap)  # original, day 2: it has drifted
    snap.birth_year, snap.birth_day_of_year = 3, 300
    assert dr.is_first_turn(snap)  # newborn's first act: it has not drifted
    snap.age_days = 2
    assert not dr.is_first_turn(snap)
