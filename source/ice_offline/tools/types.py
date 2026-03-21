from dataclasses import dataclass
from ice_offline.replay.state import State


@dataclass(frozen=True)
class Transition:
    action: int
    reward: float = 0.0
