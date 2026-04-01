from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np


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


class ValueRecordWrapper(gym.Wrapper):
    """Record per-state distribution values into info['values'].

    Behavior:
    - reset: auto record once
    - step: no auto record
    - record(): manual trigger after external learning/update
    """

    def __init__(
        self,
        env: gym.Env,
    ) -> None:
        super().__init__(env)
        self._base_env = self.env.unwrapped
        self._obs_cache: dict[tuple[int, int, int], Any] = {}
        self._observation_transforms: list[Callable[[Any], Any]] = []
        self._last_info: dict[str, Any] = {}

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._last_info = info
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_info = info
        return obs, reward, terminated, truncated, info

    def record(self, value_fn: Callable[[Any, int], float]) -> np.ndarray:
        values = self._compute_distribution_values(value_fn)
        self._last_info["values"] = values
        return values

    # ------------------------------------------------------------------
    # Value Map Build
    # ------------------------------------------------------------------
    def _compute_distribution_values(self, value_fn: Callable[[Any, int], float]) -> np.ndarray:
        width = int(self._base_env.width)
        height = int(self._base_env.height)
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
                    obs_i = self._get_cached_observation(x, y, int(d))
                    for action in MiniGridAction:
                        values[ix, iy, int(d), int(action)] = value_fn(obs_i, int(action))
        return values

    def _get_cached_observation(self, x: int, y: int, d: int) -> Any:
        key = (x, y, d)
        obs_i = self._obs_cache.get(key)
        if obs_i is None:
            obs_i = self._build_observation(x, y, d)
            self._obs_cache[key] = obs_i
        return obs_i

    def _build_observation(self, x: int, y: int, d: int) -> Any:
        if not self._observation_transforms:
            self._observation_transforms = self._build_observation_transforms()
        old_pos = tuple(self._base_env.agent_pos)
        old_dir = int(self._base_env.agent_dir)
        try:
            self._base_env.agent_pos = (x, y)
            self._base_env.agent_dir = d
            obs = self._base_env.gen_obs()
            for transform in self._observation_transforms:
                obs = transform(obs)
        finally:
            self._base_env.agent_pos = old_pos
            self._base_env.agent_dir = old_dir
        return obs

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


def ensure_record_wrapper(env: gym.Env) -> gym.Env:
    current: gym.Env = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, ValueRecordWrapper):
            return env
        current = current.env
    return ValueRecordWrapper(env)
