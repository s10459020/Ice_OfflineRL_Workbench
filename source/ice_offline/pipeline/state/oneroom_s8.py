from dataclasses import dataclass
from typing import Any

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj

from ice_offline.pipeline.state._spec import State, StateIO


@dataclass(frozen=True)
class OneroomS8State(State):
    mission: str
    agent_pos: tuple[int, int]
    agent_dir: int
    grid: np.ndarray
    carrying: tuple[int, int, int] | None

    def serialize(self) -> dict[str, Any]:
        return {
            "mission": self.mission.encode("utf-8"),
            "agent_pos": np.asarray(self.agent_pos, dtype=np.int16),
            "agent_dir": self.agent_dir,
            "grid": np.asarray(self.grid, dtype=np.int16),
            "carrying": np.asarray(self.carrying or (0, 0, 0), dtype=np.int16),
        }

    @classmethod
    def from_serialized(cls, payload: dict[str, Any]) -> "OneroomS8State":
        encoded = tuple(payload["carrying"])
        carrying = None if encoded == (0, 0, 0) else encoded
        agent_pos_xy = tuple(payload["agent_pos"])
        mission_raw = payload["mission"]
        return cls(
            mission=mission_raw.decode("utf-8"),
            agent_pos=agent_pos_xy,
            agent_dir=payload["agent_dir"],
            grid=np.asarray(payload["grid"], dtype=np.int16),
            carrying=carrying,
        )


class OneroomS8StateIO(StateIO):
    """State IO adapter for OneRoomS8-style MiniGrid environments."""

    def __init__(self, env: Any) -> None:
        self._env = env

    def get_state(self) -> OneroomS8State:
        base = self._env.unwrapped
        carrying = None if base.carrying is None else tuple(base.carrying.encode())
        return OneroomS8State(
            mission=base.mission,
            agent_pos=base.agent_pos,
            agent_dir=base.agent_dir,
            grid=np.asarray(base.grid.encode(), dtype=np.int8),
            carrying=carrying,
        )

    def set_state(self, state: OneroomS8State) -> None:
        base = self._env.unwrapped
        decoded = Grid.decode(np.asarray(state.grid, dtype=np.uint8))
        carrying = None
        if state.carrying is not None:
            carrying = WorldObj.decode(*state.carrying)

        base.grid = decoded[0]
        base.agent_pos = state.agent_pos
        base.agent_dir = state.agent_dir
        base.mission = state.mission
        base.carrying = carrying
