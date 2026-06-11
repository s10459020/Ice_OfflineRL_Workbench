from pathlib import Path
from typing import Any, Type

import gymnasium as gym

from ice_offline.store.state._spec import State, StateIO
from ice_offline.store.state.op_dataset import StateDataset


class StateCollectWrapper(gym.Wrapper):
    # ====================
    # Init
    # ====================
    def __init__(self, env: gym.Env, state_cls: Type[State], state_io_cls: Type[StateIO]) -> None:
        super().__init__(env)
        self._state_cls = state_cls
        self._state_io = state_io_cls(env)
        self._episodes: list[list[dict]] = []
        self._episode: list[dict] | None = None

    # ====================
    # Public API
    # ====================
    def save(self, path: Path) -> StateDataset:
        self._end_episode()
        return StateDataset.write(
            path=path.with_name("state_data.hdf5"),
            state_cls=self._state_cls,
            episodes=self._episodes,
        )

    # ====================
    # Overwrite
    # ====================
    def reset(self, *args: Any, **kwargs: Any):
        obs, info = self.env.reset(*args, **kwargs)
        state = self._state_io.get_state()
        self._end_episode()
        self._episode = [state.serialize()]
        info["state"] = state
        return obs, info

    def step(self, *args: Any, **kwargs: Any):
        obs, reward, terminated, truncated, info = self.env.step(*args, **kwargs)
        state = self._state_io.get_state()
        self._episode.append(state.serialize())
        info["state"] = state
        return obs, reward, terminated, truncated, info

    # ====================
    # Private
    # ====================
    def _end_episode(self) -> None:
        if self._episode:
            self._episodes.append(self._episode)

