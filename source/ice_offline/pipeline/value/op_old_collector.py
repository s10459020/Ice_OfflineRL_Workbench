
from pathlib import Path
from collections.abc import Callable
from enum import IntEnum
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
from ice_offline.tools.paths import minari_root


class MiniGridDirection(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


class MiniGridAction(IntEnum):
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    PICKUP = 3


class OldValueCollector(gym.Wrapper):
    """Collect episode-wise values and save to Minari dataset folder.

    HDF5 format:
    - episode_{index}/values -> ndarray with shape [T+1, ...]
    """

    # ====================
    # Init
    # ====================
    def __init__(self, env: Any, value_fn: Callable[[Any, int], float]) -> None:
        super().__init__(env)
        self._value_fn = value_fn
        self._base_env = self.env.unwrapped
        self._episodes: list[list[np.ndarray]] = []
        self._current: list[np.ndarray] | None = None
        self._last_values: np.ndarray | None = None
        self._obs_cache: dict[tuple[int, int, int], Any] = {}
        self._last_carrying_obj: Any = None
        self._observation_transforms: list[Callable[[Any], Any]] = []

    # ====================
    # Public API
    # ====================
    def record(self) -> np.ndarray:
        """Capture one value snapshot at current env/agent context."""
        values = self._compute_values(self._value_fn)
        self._current.append(values)
        self._last_values = values
        return values

    def get_last(self) -> np.ndarray | None:
        """Return the latest recorded value tensor."""
        return self._last_values

    def save(self, dataset_id: str) -> Path:
        """Persist all buffered episodes into value_data.hdf5."""
        self._end_episode()

        out_path = self._resolve_value_path(dataset_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(out_path, "w") as f:
            for ep_idx, seq in enumerate(self._episodes):
                stacked = np.stack(seq, axis=0)
                ep_group = f.require_group(f"episode_{ep_idx}")
                ep_group.create_dataset("values", data=stacked)
        return out_path

    # ====================
    # gym.Wrapper Overrides
    # ====================
    def reset(self, *args: Any, **kwargs: Any):
        """Start a new episode and clear value collection cache."""
        self._end_episode()
        self._obs_cache.clear()
        self._last_carrying_obj = None
        self._current = []
        self._last_values = None
        return self.env.reset(*args, **kwargs)

    # ====================
    # Internal
    # ====================
    def _compute_values(self, value_fn: Callable[[Any, int], float]) -> np.ndarray:
        #carrying_obj = self._base_env.carrying
        #if self._last_carrying_obj is not carrying_obj:
        #    self._obs_cache.clear()
        #    self._last_carrying_obj = carrying_obj
 
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
            if not self._observation_transforms:
                self._observation_transforms = self._build_observation_transforms()
            old_pos = tuple(self._base_env.agent_pos)
            old_dir = self._base_env.agent_dir
            try:
                self._base_env.agent_pos = (x, y)
                self._base_env.agent_dir = d
                obs_i = self._base_env.gen_obs()
                for transform in self._observation_transforms:
                    obs_i = transform(obs_i)
            finally:
                self._base_env.agent_pos = old_pos
                self._base_env.agent_dir = old_dir
            self._obs_cache[key] = obs_i
        return obs_i

    def _build_observation_transforms(self) -> list[Callable[[Any], Any]]:
        wrappers: list[gym.Wrapper] = []
        current: gym.Env = self.env
        while isinstance(current, gym.Wrapper):
            wrappers.append(current)
            current = current.env
        transforms: list[Callable[[Any], Any]] = []
        for wrapper in reversed(wrappers):
            if isinstance(wrapper, gym.ObservationWrapper):
                transforms.append(wrapper.observation)
        return transforms

    # ====================
    # Path
    # ====================
    def _resolve_value_path(self, dataset_id: str) -> Path:
        base = minari_root()
        return base / dataset_id / "data" / "value_data.hdf5"

    def _end_episode(self) -> None:
        if self._current:
            self._episodes.append(self._current)
