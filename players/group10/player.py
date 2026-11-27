import random

from core.action import Action
from core.depth import Depth
from core.player import Player
from core.seasons import Season
from core.snapshot import JellyfishSnapshot
from players.group10 import dead_reckoning


class Player10(Player):
    """Group 10's jellyfish.

    Picks depth by the following rules:

    1. Predator zone: ride the fastest current to get out.
    2. Feeding zone in spring or summer: slow down to eat.
    3. Breeding zone in fall or winter with offspring left to spawn: slow down.
    4. Anywhere else, cruise until it drifts into a zone:
       a. go fast if there is a zone wanted to be in (e.g. feeding zone during summer)
       b. otherwise rng pick between fast and cruise to explore.

    Every day it also updates its dead-reckoning chart in memory (own
    position plus feeding / breeding centre estimates, see
    `dead_reckoning.py`). The rules above do not use the chart yet.
    """

    ESCAPE = Depth.SURFACE  # 40 km/day
    FAST_CRUISE = Depth.SURFACE  # 40 km/day
    CRUISE = Depth.SHALLOW  # 20 km/day
    SLOW_CRUISE = Depth.MEDIUM  # 10 km/day
    LINGER = Depth.DEEP  # 5 km/day

    # Slow down when the zone is open or about to open.
    FEEDING_SEASONS = (Season.SPRING, Season.SUMMER)
    BREEDING_SEASONS = (Season.FALL, Season.WINTER)

    def __init__(self, memory_size: int, lifetime_days: int, rng: random.Random):
        super().__init__(memory_size, lifetime_days, rng)
        # None when memory is too small; the rules below then run on senses only.
        self.layout = dead_reckoning.layout_for(memory_size)

    def act(self, snapshot: JellyfishSnapshot) -> Action:
        if self.layout is not None:
            dead_reckoning.update(snapshot, self.layout)

        if snapshot.in_predator_zone:
            return Action(self.ESCAPE)

        if snapshot.in_feeding_zone and snapshot.season in self.FEEDING_SEASONS:
            return Action(self.LINGER)

        can_reproduce = snapshot.reproduction_capacity > 0
        if (
            snapshot.in_breeding_zone
            and can_reproduce
            and snapshot.season in self.BREEDING_SEASONS
        ):
            return Action(self.LINGER)

        # If not in a zone it wants to stay in.
        missing_feeding = snapshot.season == Season.SUMMER
        missing_breeding = snapshot.season == Season.WINTER and can_reproduce
        if missing_feeding or missing_breeding:
            return Action(self.FAST_CRUISE)

        # Else randomly pick FAST or CRUISE for varied exploration
        return Action(self.rng.choice((self.FAST_CRUISE, self.CRUISE)))
