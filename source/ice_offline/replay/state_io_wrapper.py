from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj

from ice_offline.replay.state import State


class StateIOWrapper(gym.Wrapper):
    """Provide unified state I/O methods for wrapped envs."""

    def get_state(self) -> State:
        base = self.env.unwrapped
        x, y = base.agent_pos

        carrying = None
        if base.carrying is not None:
            carrying = tuple(int(v) for v in base.carrying.encode())
            
        return State(
            mission=base.mission,
            agent_pos=(x, y),
            agent_dir=base.agent_dir,
            grid=np.asarray(base.grid.encode(), dtype=np.uint8),
            carrying=carrying,
        )

    def set_state(self, state: State) -> None:
        base = self.env.unwrapped
        decoded = Grid.decode(np.asarray(state.grid, dtype=np.uint8))
        
        carrying = None
        if state.carrying is None:
            carrying = WorldObj.decode(*tuple(int(v) for v in state.carrying))

        base.grid = decoded[0] if isinstance(decoded, tuple) else decoded
        base.agent_pos = tuple(state.agent_pos)
        base.agent_dir = int(state.agent_dir)
        base.mission = state.mission
        base.carrying = carrying

def ensure_state_io(env: gym.Env) -> gym.Env:
    current: Any = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, StateIOWrapper):
            return env
        current = current.env
    return StateIOWrapper(env)
