from typing import Any
from types import MethodType

import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj

from ice_offline.env.model import State


class StateIOWrapper(gym.Wrapper):
    """Inject get_state/set_state into env and keep wrapper-chain compatibility."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not (callable(getattr(self.env, "get_state", None)) and callable(getattr(self.env, "set_state", None))):
            self.env.get_state = MethodType(_get_state, self.env)
            self.env.set_state = MethodType(_set_state, self.env)

def ensure_state_io(env: gym.Env) -> gym.Env:
    current: Any = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, StateIOWrapper):
            return env
        current = current.env
    return StateIOWrapper(env)


def _get_state(env: gym.Env) -> State:
    base = env.unwrapped
    carrying = None if base.carrying is None else tuple(base.carrying.encode())
    return State(
        mission=base.mission,
        agent_pos=base.agent_pos,
        agent_dir=base.agent_dir,
        grid=np.asarray(base.grid.encode(), dtype=np.int8),
        carrying=carrying,
    )


def _set_state(env: gym.Env, state: State) -> None:
    base = env.unwrapped
    decoded = Grid.decode(np.asarray(state.grid, dtype=np.uint8))
    carrying = None
    if state.carrying is not None:
        carrying = WorldObj.decode(*state.carrying)

    base.grid = decoded[0]
    base.agent_pos = state.agent_pos
    base.agent_dir = state.agent_dir
    base.mission = state.mission
    base.carrying = carrying
