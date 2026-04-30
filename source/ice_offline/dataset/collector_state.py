
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np

from ice_offline.env.common import StateIOWrapper
from ice_offline.data import State
from ice_offline.tools.paths import minari_root

class StateCollector(gym.Wrapper):
    """Collect episode-wise states and save to Minari dataset folder.

    HDF5 format:
    - episode_{index}/{field} -> ndarray with shape [T+1, ...]
    """

    # ====================
    # Init
    # ====================
    def __init__(self, env: Any) -> None:
        self._state_io = StateIOWrapper(env)
        super().__init__(self._state_io)
        self._episodes: list[list[dict[str, Any]]] = []
        self._current: list[dict[str, Any]] | None = None
        self._last_state: State | None = None

    # ====================
    # Public API
    # ====================
    def get_last(self) -> State | None:
        """Return the latest captured state in current rollout."""
        return self._last_state

    def save(self, dataset_id: str) -> Path:
        """Persist all buffered episodes into state_data.hdf5."""
        self._end_episode()

        out_path = self._resolve_state_path(dataset_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(out_path, "w") as f:
            for ep_idx, seq in enumerate(self._episodes):
                ep_group = f.require_group(f"episode_{ep_idx}")
                keys = seq[0].keys()
                for key in keys:
                    values = [item[key] for item in seq]
                    ep_group.create_dataset(key, data=np.asarray(values))
        return out_path

    # ====================
    # gym.Wrapper Overrides
    # ====================
    def reset(self, *args: Any, **kwargs: Any):
        """Start a new episode and capture the initial state (t=0)."""
        self._end_episode()
        
        obs, info = self.env.reset(*args, **kwargs)
        state = self._state_io.get_state()
        self._last_state = state
        
        self._current = [state.serialize()]
        return obs, info

    def step(self, *args: Any, **kwargs: Any):
        """Advance env one step and append the resulting state."""
        obs, reward, terminated, truncated, info = self.env.step(*args, **kwargs)
        state = self._state_io.get_state()
        self._last_state = state

        self._current.append(state.serialize())
        return obs, reward, terminated, truncated, info

    # ====================
    # Internal
    # ====================
    def _resolve_state_path(self, dataset_id: str) -> Path:
        base = minari_root()
        return base / dataset_id / "data" / "state_data.hdf5"

    def _end_episode(self) -> None:
        if self._current:
            self._episodes.append(self._current)
