from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj

from ice_offline.env.model import State


class StateIOWrapper(gym.Wrapper):
    """Provide unified state I/O methods for wrapped envs."""

    def get_state(self) -> State:
        base = self.env.unwrapped
        carrying = None if base.carrying is None else tuple(base.carrying.encode())

        return State(
            mission=base.mission,
            agent_pos=base.agent_pos,
            agent_dir=base.agent_dir,
            grid=np.asarray(base.grid.encode(), dtype=np.int8),
            carrying=carrying,
        )

    def set_state(self, state: State) -> None:
        base = self.env.unwrapped
        decoded = Grid.decode(np.asarray(state.grid, dtype=np.uint8))

        carrying = None
        if state.carrying is not None:
            carrying = WorldObj.decode(*state.carrying)

        base.grid = decoded[0]
        base.agent_pos = state.agent_pos
        base.agent_dir = state.agent_dir
        base.mission = state.mission
        base.carrying = carrying

def ensure_state_io(env: gym.Env) -> gym.Env:
    current: Any = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, StateIOWrapper):
            return env
        current = current.env
    return StateIOWrapper(env)
