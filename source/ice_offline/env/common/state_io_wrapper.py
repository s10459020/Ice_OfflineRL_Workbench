import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj

from ice_offline.data.state import State


class StateIOWrapper(gym.Wrapper):
    """Provide get_state/set_state only on this wrapper layer."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        
    def get_state(self) -> State:
        base = self.unwrapped
        carrying = None if base.carrying is None else tuple(base.carrying.encode())
        return State(
            mission=base.mission,
            agent_pos=base.agent_pos,
            agent_dir=base.agent_dir,
            grid=np.asarray(base.grid.encode(), dtype=np.int8),
            carrying=carrying
        )

    def set_state(self, state: State) -> None:
        base = self.unwrapped
        decoded = Grid.decode(np.asarray(state.grid, dtype=np.uint8))
        carrying = None
        if state.carrying is not None:
            carrying = WorldObj.decode(*state.carrying)

        base.grid = decoded[0]
        base.agent_pos = state.agent_pos
        base.agent_dir = state.agent_dir
        base.mission = state.mission
        base.carrying = carrying
