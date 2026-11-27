import random
from math import cos, sin, tau

from core.action import Action
from core.constants import OCEAN_SIZE
from core.depth import Depth
from core.player import Player
from core.seasons import Season
from core.snapshot import JellyfishSnapshot
from players.group10 import dead_reckoning, navigation


class Player10(Player):
    """Group 10's jellyfish.

    With 8+ bytes of memory it navigates by its dead-reckoning chart
    (`dead_reckoning.py`) and the stored current headings (`navigation.py`):

    1. Original jellyfish spend their first days visiting each depth once to
       learn the four current headings. Offspring inherit them.
    2. Pick the goal zone for the season: breeding while it still has
       offspring to spawn this fall / winter, feeding otherwise (it waits at
       the feeding zone for summer).
    3. Inside the goal zone: hold position at the stored hold point (first
       the estimated centre). If that spot is in the predator zone, the hold
       point moves. Outside: ride the fastest mix of currents toward it.
    4. Goal zone never seen: lawnmower search (`navigation.sweep`).

    With less memory it falls back to sense-only rules (`_sense_only`).
    """

    ESCAPE = Depth.SURFACE  # 40 km/day
    FAST_CRUISE = Depth.SURFACE  # 40 km/day
    CRUISE = Depth.SHALLOW  # 20 km/day
    LINGER = Depth.DEEP  # 5 km/day

    # How far the hold point jumps when it turns out to be in the predator zone.
    RELOCATE_KM = 40.0

    # Slow down when the zone is open or about to open.
    FEEDING_SEASONS = (Season.SPRING, Season.SUMMER)
    BREEDING_SEASONS = (Season.FALL, Season.WINTER)

    def __init__(self, memory_size: int, lifetime_days: int, rng: random.Random):
        super().__init__(memory_size, lifetime_days, rng)
        # Both depend only on the memory size, never on a particular jellyfish.
        self.layout = dead_reckoning.layout_for(memory_size)
        self.headings = (
            navigation.heading_store_for(memory_size, self.layout.num_bytes)
            if self.layout is not None
            else None
        )

    def act(self, snapshot: JellyfishSnapshot) -> Action:
        if self.layout is None:
            return self._sense_only(snapshot)
        chart = dead_reckoning.update(snapshot, self.layout)
        if self.headings is None:
            return self._sense_only(snapshot)

        unseen = navigation.learn(snapshot, self.headings)
        if unseen is not None:
            return Action(unseen)
        vels = navigation.known_velocities(
            snapshot, self.headings.read(snapshot.memory)
        )

        zone, inside = self._goal(snapshot)
        vector = chart.vector_to(zone, self.layout)
        if vector is None:
            return Action(self._search(snapshot, vels))
        if inside:
            if snapshot.in_predator_zone and self.RELOCATE_KM > 0:
                vector = self._relocate(snapshot, chart, zone)
            return Action(navigation.hold(vector, vels))
        depth = navigation.steer(vector, vels)
        return Action(depth if depth is not None else self._search(snapshot, vels))

    def _relocate(
        self, s: JellyfishSnapshot, chart: dead_reckoning.Chart, zone: str
    ) -> tuple[float, float]:
        """Move the stored hold point: this spot is inside the predator zone.

        We cannot afford to remember where the predator zone is, so the hold
        point takes a random step. The in/out corrections keep it inside the
        goal zone, and once it lands on a safe spot it stops moving. Offspring
        inherit the safe spot.
        """
        cx, cy = chart.zones[zone]
        angle = self.rng.uniform(0, tau)
        cx += self.RELOCATE_KM * cos(angle)
        cy += self.RELOCATE_KM * sin(angle)
        chart.zones[zone] = (cx % OCEAN_SIZE, cy % OCEAN_SIZE)
        dead_reckoning.encode(chart, s.memory, self.layout)
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
