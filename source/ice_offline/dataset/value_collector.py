from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import h5py
import numpy as np

from ice_offline.env.replay.value_record_wrapper import MiniGridAction, MiniGridDirection


class ValueCollector(gym.Wrapper):
    """Collect episode-wise values and save to Minari dataset folder.

    HDF5 format:
    - episode_{index}/values -> ndarray with shape [T+1, ...]
    """

    # ====================
    # Init
    # ====================
    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self._base_env = self.env.unwrapped
        self._episodes: list[list[np.ndarray]] = []
        self._current: list[np.ndarray] | None = None
        self._obs_cache: dict[tuple[int, int, int], Any] = {}

    # ====================
    # Public API
    # ====================
    def reset(self, value_fn: Callable[[Any, int], float], *args: Any, **kwargs: Any):
        obs, info = self.env.reset(*args, **kwargs)
        if self._current:
            self._episodes.append(self._current)
        values = self._compute_values(value_fn)
        self._current = [values]
        return obs, info

    def step(self, value_fn: Callable[[Any, int], float], *args: Any, **kwargs: Any):
        obs, reward, terminated, truncated, info = self.env.step(*args, **kwargs)
        values = self._compute_values(value_fn)
        self._current.append(values)
        return obs, reward, terminated, truncated, info

    def save(self, dataset_id: str) -> Path:
        if self._current:
            self._episodes.append(self._current)

        out_path = self._resolve_value_path(dataset_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(out_path, "w") as f:
            for ep_idx, seq in enumerate(self._episodes):
                stacked = np.stack(seq, axis=0)
                ep_group = f.require_group(f"episode_{ep_idx}")
                ep_group.create_dataset("values", data=stacked)
        return out_path

    # ====================
    # Internal
    # ====================
    def _compute_values(self, value_fn: Callable[[Any, int], float]) -> np.ndarray:
        width = self._base_env.width
        height = self._base_env.height
        inner_w = max(0, width - 2)
        inner_h = max(0, height - 2)
        values = np.zeros(
            (inner_w, inner_h, len(MiniGridDirection), len(MiniGridAction)),
            dtype=np.float32,
        )
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                ix = x - 1
                iy = y - 1
                for d in MiniGridDirection:
                    obs_i = self._get_cached_observation(x, y, d)
                    for action in MiniGridAction:
                        values[ix, iy, d, action] = value_fn(obs_i, action)
        return values

    def _get_cached_observation(self, x: int, y: int, d: int) -> Any:
        key = (x, y, d)
        obs_i = self._obs_cache.get(key)
        if obs_i is None:
            old_pos = tuple(self._base_env.agent_pos)
            old_dir = self._base_env.agent_dir
            try:
                self._base_env.agent_pos = (x, y)
                self._base_env.agent_dir = d
                obs_i = self._base_env.gen_obs()
            finally:
                self._base_env.agent_pos = old_pos
                self._base_env.agent_dir = old_dir
            self._obs_cache[key] = obs_i
        return obs_i

    # ====================
    # Path
    # ====================
    def _resolve_value_path(self, dataset_id: str) -> Path:
        root = os.getenv("MINARI_DATASETS_PATH")
        if root:
            base = Path(root)
        else:
            base = Path.home() / ".minari" / "datasets"
        return base / dataset_id / "data" / "value_data.hdf5"
