from typing import Any

import gymnasium as gym
import numpy as np

from ice_offline.store.probe.op_collector import ProbeInterface


class HopperOodActionProbe(ProbeInterface):
    def __init__(self, sample_count: int = 100) -> None:
        self._sample_count = sample_count
        self._env: gym.Env | None = None

    def set_env(self, env: gym.Env) -> None:
        self._env = env
        self._action_dim = int(np.prod(env.action_space.shape))
        self._repeat = self._action_dim * self._sample_count
        self._base_action = np.zeros(self._action_dim, dtype=np.float32)
        self._samples = self._build_samples()

    def reset(self, state: Any) -> None:
        self._base_action = np.zeros(self._action_dim, dtype=np.float32)

    def step(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        self._base_action = np.asarray(action, dtype=np.float32).reshape(self._action_dim)

    def get_probes(self, observation: Any) -> tuple[np.ndarray, np.ndarray]:
        observation_array = np.asarray(observation, dtype=np.float32)
        segments: list[np.ndarray] = []
        for action_index in range(self._action_dim):
            actions = np.repeat(self._base_action[None, :], self._sample_count, axis=0)
            actions[:, action_index] = self._samples[action_index]
            segments.append(actions)
        actions = np.concatenate(segments, axis=0)
        observations = np.repeat(observation_array[None, :], actions.shape[0], axis=0)
        return observations, actions

    def _build_samples(self) -> np.ndarray:
        low = np.asarray(self._env.action_space.low, dtype=np.float32).reshape(self._action_dim)
        high = np.asarray(self._env.action_space.high, dtype=np.float32).reshape(self._action_dim)
        return np.asarray([
            np.linspace(
                float(low[action_index]),
                float(high[action_index]),
                self._sample_count,
                dtype=np.float32,
            )
            for action_index in range(self._action_dim)
        ], dtype=np.float32)
