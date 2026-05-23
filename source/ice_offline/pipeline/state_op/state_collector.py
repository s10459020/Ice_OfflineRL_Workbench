from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np

from ice_offline.pipeline.state._spec import StateIO
from ice_offline.tools.paths import minari_root


class StateCollector(gym.Wrapper):
    """Collect episode-wise states and save to Minari dataset folder."""

    # ====================
    # Init
    # ====================
    def __init__(self, env: gym.Env, state_io: StateIO) -> None:
        super().__init__(env)
        self._state_io = state_io
        self._episodes: list[list[dict]] = []
        self._episode: list[dict] | None = None

    # ====================
    # Public API
    # ====================
    def save(self, dataset_id: str) -> Path:
        self._end_episode()
        out_path = self._resolve_state_path(dataset_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(out_path, "w") as h5_file:
            for ep_idx, seq in enumerate(self._episodes):
                ep_group = h5_file.require_group(f"episode_{ep_idx}")
                keys = seq[0].keys()
                for key in keys:
                    values = [item[key] for item in seq]
                    ep_group.create_dataset(key, data=np.asarray(values))
        return out_path

    # ====================
    # Override
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
    def _resolve_state_path(self, dataset_id: str) -> Path:
        base = minari_root()
        return base / dataset_id / "data" / "state_data.hdf5"

    def _end_episode(self) -> None:
        if self._episode:
            self._episodes.append(self._episode)
