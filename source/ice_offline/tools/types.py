from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class State:
    mission: str
    agent_pos: tuple[int, int]
    agent_dir: int
    grid: np.ndarray
    carrying: dict[str, Any] | None


@dataclass(frozen=True)
class Transition:
    action: int
    reward: float = 0.0
