from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class State:
    mission: str
    agent_pos: tuple[int, int]
    agent_dir: int
    grid: np.ndarray
    carrying: tuple[int, int, int] | None

    def serialize(self) -> dict[str, Any]:
        return {
            "mission": str(self.mission),
            "agent_pos": np.asarray(
                [int(self.agent_pos[0]), int(self.agent_pos[1])],
                dtype=np.int16,
            ),
            "agent_dir": int(self.agent_dir),
            "grid": np.asarray(self.grid, dtype=np.int16),
            "carrying": np.asarray(self.carrying or (0, 0, 0), dtype=np.int16),
        }

    @classmethod
    def from_serialized(cls, payload: dict[str, Any]) -> "State":
        encoded = tuple(int(v) for v in np.asarray(payload["carrying"]).tolist())
        carrying = None if encoded == (0, 0, 0) else encoded

        agent_pos = payload["agent_pos"]
        agent_pos_xy = tuple(int(v) for v in np.asarray(agent_pos).tolist())

        return cls(
            mission=str(payload["mission"]),
            agent_pos=agent_pos_xy,
            agent_dir=int(payload["agent_dir"]),
            grid=np.asarray(payload["grid"], dtype=np.int16),
            carrying=carrying,
        )
