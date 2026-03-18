
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AgentState:
    mission: str
    agent_pos: tuple[int, int]
    agent_dir: int
    grid: np.ndarray
    carrying: dict[str, Any] | None
