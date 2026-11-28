import random
from math import cos, hypot, sin, tau

from core.action import Action
from core.constants import OCEAN_SIZE
from core.depth import Depth
from core.player import Player
from core.seasons import Season
from core.snapshot import JellyfishSnapshot
from players.group10 import dead_reckoning, navigation


class Player10(Player):
    """Group 10's jellyfish.

    With 8+ bytes of memory it keeps a chart in memory (`dead_reckoning.py`:
    position, zone points, predator centre, current headings) and navigates
    by it (`navigation.py`):

    1. Original jellyfish spend their first days visiting each depth once to
       learn the four current headings. Offspring inherit them.
    2. Pick the goal zone for the season: breeding while it still has
       offspring to spawn this fall / winter, feeding otherwise (it waits at
       the feeding zone for summer).
    3. Inside the goal zone: hold position at the stored hold point (first
       the estimated centre). If that spot is in the predator zone, the hold
       point moves away from it. Outside: ride the mix of currents toward it
       that best trades travel time against days in the predator zone.
    4. Goal zone never seen: lawnmower search (`navigation.sweep`).

    With less memory it falls back to sense-only rules (`_sense_only`).
    """

    ESCAPE = Depth.SURFACE  # 40 km/day
    FAST_CRUISE = Depth.SURFACE  # 40 km/day
    CRUISE = Depth.SHALLOW  # 20 km/day
    LINGER = Depth.DEEP  # 5 km/day

    # How far the hold point jumps when it turns out to be in the predator zone.
    RELOCATE_KM = 40.0

    # Chance per day of riding a random depth instead of the planned one.
    # 0.02 cost no speed over seeds 1-48; 0.05+ delayed 10x growth by a year.
    EXPLORE_PROB = 0.02

    # Slow down when the zone is open or about to open.
    FEEDING_SEASONS = (Season.SPRING, Season.SUMMER)
    BREEDING_SEASONS = (Season.FALL, Season.WINTER)

    def __init__(self, memory_size: int, lifetime_days: int, rng: random.Random):
        super().__init__(memory_size, lifetime_days, rng)
        # Depends only on the memory size, never on a particular jellyfish.
        self.layout = dead_reckoning.layout_for(memory_size)

    def act(self, snapshot: JellyfishSnapshot) -> Action:
        if self.layout is None:
            return self._sense_only(snapshot)
        chart = dead_reckoning.decode(snapshot.memory, self.layout)
        dead_reckoning.update(chart, snapshot, self.layout)
        action = self._navigate(snapshot, chart)
        dead_reckoning.encode(chart, snapshot.memory, self.layout)
        return action

    def _navigate(self, s: JellyfishSnapshot, chart: dead_reckoning.Chart) -> Action:
        # An original rides one depth deeper each day for its first four
        # days, recording each current's heading. Offspring inherit them.
        known = dead_reckoning.headings_known(s)
        if known < 4:
            chart.set_heading(s.depth, s.heading, self.layout)
            if known < 3:
                return Action(Depth(known + 1))
        # Bet hedging: a parent and its offspring share memory and position,
        # so without this a whole lineage makes every decision together, and
        # one wrong chart or route dooms it all. An occasional random depth
        # spreads them out. The chart stays right: it integrates the actual
        # drift (`s.velocity`), not the depth we chose.
        if self.rng.random() < self.EXPLORE_PROB and not s.in_predator_zone:
            return Action(self.rng.choice(tuple(Depth)))
        vels = self._velocities(s, chart)

        zone, inside = self._goal(s)
        vector = chart.vector_to(zone, self.layout)
        if vector is None:
            return Action(self._search(s, vels))
        predator = chart.vector_to(dead_reckoning.PREDATOR, self.layout)
        if inside:
            if s.in_predator_zone and self.RELOCATE_KM > 0:
                vector = self._relocate(chart, zone, predator)
            return Action(navigation.hold(vector, vels))
        day_phase = navigation.phase((s.year - 1) * 360 + s.day_of_year)
        depth = navigation.steer(vector, vels, predator, day_phase, s.depth)
        return Action(depth if depth is not None else self._search(s, vels))

    def _velocities(self, s: JellyfishSnapshot, chart) -> list[tuple[float, float]]:
        """Velocity of every current; today's is exact, the rest from memory."""
        vels = chart.velocities(self.layout)
        vels[s.depth] = s.velocity
        return vels

    def _relocate(
        self,
        chart: dead_reckoning.Chart,
        zone: str,
        predator: tuple[float, float] | None,
    ) -> tuple[float, float]:
        """Move the stored hold point: this spot is inside the predator zone.

        The point steps away from the estimated predator centre (randomly if
        that is unknown or right here). The in/out corrections keep it inside
        the goal zone, and once it lands on a safe spot it stops moving.
        Offspring inherit the safe spot.
        """
        hx, hy = chart.vector_to(zone, self.layout)
        dx, dy = (hx - predator[0], hy - predator[1]) if predator else (0.0, 0.0)
        norm = hypot(dx, dy)
        if norm < 1.0:
            angle = self.rng.uniform(0, tau)
            dx, dy, norm = cos(angle), sin(angle), 1.0
        cx, cy = chart.zones[zone]
        cx += self.RELOCATE_KM * dx / norm
        cy += self.RELOCATE_KM * dy / norm
        chart.zones[zone] = (cx % OCEAN_SIZE, cy % OCEAN_SIZE)
        return chart.vector_to(zone, self.layout)

    def _goal(self, s: JellyfishSnapshot) -> tuple[str, bool]:
        if s.reproduction_capacity > 0 and s.season in self.BREEDING_SEASONS:
            return dead_reckoning.BREEDING, s.in_breeding_zone
        return dead_reckoning.FEEDING, s.in_feeding_zone

    def _search(self, s: JellyfishSnapshot, vels) -> Depth:
        if s.in_predator_zone:
            return self.ESCAPE
        day_index = (s.year - 1) * 360 + s.day_of_year
        return navigation.sweep(day_index, vels)

    def _sense_only(self, s: JellyfishSnapshot) -> Action:
        """The original memory-free rules, for memory below 8 bytes."""
        if s.in_predator_zone:
            return Action(self.ESCAPE)

        if s.in_feeding_zone and s.season in self.FEEDING_SEASONS:
            return Action(self.LINGER)

        can_reproduce = s.reproduction_capacity > 0
        if s.in_breeding_zone and can_reproduce and s.season in self.BREEDING_SEASONS:
            return Action(self.LINGER)

        # If not in a zone it wants to stay in.
        missing_feeding = s.season == Season.SUMMER
        missing_breeding = s.season == Season.WINTER and can_reproduce
        if missing_feeding or missing_breeding:
            return Action(self.FAST_CRUISE)

        # Else randomly pick FAST or CRUISE for varied exploration
        return Action(self.rng.choice((self.FAST_CRUISE, self.CRUISE)))
