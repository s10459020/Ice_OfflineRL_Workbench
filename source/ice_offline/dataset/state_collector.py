from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np

from ice_offline.env.common import StateIOWrapper

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

    # ====================
    # Public API
    # ====================
    def reset(self, *args: Any, **kwargs: Any):
        obs, info = self.env.reset(*args, **kwargs)
        if self._current:
            self._episodes.append(self._current)
        state = self._state_io.get_state().serialize()
        self._current = [state]
        return obs, info

    def step(self, *args: Any, **kwargs: Any):
        obs, reward, terminated, truncated, info = self.env.step(*args, **kwargs)
        state = self._state_io.get_state().serialize()
        self._current.append(state)
        return obs, reward, terminated, truncated, info

    def save(self, dataset_id: str) -> Path:
        if self._current:
            self._episodes.append(self._current)

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
    # Internal
    # ====================
    def _resolve_state_path(self, dataset_id: str) -> Path:
        root = os.getenv("MINARI_DATASETS_PATH")
        if root:
            base = Path(root)
        else:
            base = Path.home() / ".minari" / "datasets"
        return base / dataset_id / "data" / "state_data.hdf5"
