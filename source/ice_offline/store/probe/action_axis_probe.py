from typing import Any

import gymnasium as gym
import numpy as np

from ice_offline.store.probe.op_collector import ProbeInterface


class ActionAxisProbe(ProbeInterface):
    def __init__(self, sample_count: int = 100) -> None:
        self._sample_count = sample_count
        self._env: gym.Env | None = None

    def set_env(self, env: gym.Env) -> None:
        self._env = env
        self._action_shape = tuple(env.action_space.shape)
        self._action_dim = int(np.prod(self._action_shape))
        self._base_action = np.zeros(self._action_dim, dtype=np.float32)
        self._samples = self._build_samples()

    def reset(self, state: Any) -> None:
        self._base_action = np.zeros(self._action_dim, dtype=np.float32)

    def step(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        self._base_action = np.asarray(action, dtype=np.float32).reshape(self._action_dim)

    def get_probes(self, observation: Any) -> tuple[np.ndarray, np.ndarray]:
        observation_array = np.asarray(observation, dtype=np.float32)
        actions_per_axis: list[np.ndarray] = []
        for action_index in range(self._action_dim):
            actions = np.repeat(self._base_action[None, :], self._sample_count, axis=0)
            actions[:, action_index] = self._samples[action_index]
            actions_per_axis.append(actions.reshape(self._sample_count, *self._action_shape))
        actions = np.concatenate(actions_per_axis, axis=0)
        observations = np.repeat(observation_array[None, :], actions.shape[0], axis=0)
        return observations, actions

    def _build_samples(self) -> np.ndarray:
        low = np.asarray(self._env.action_space.low, dtype=np.float32).reshape(self._action_dim)
        high = np.asarray(self._env.action_space.high, dtype=np.float32).reshape(self._action_dim)
        return np.asarray(
            [
                np.linspace(
                    float(low[action_index]),
                    float(high[action_index]),
                    self._sample_count,
                    dtype=np.float32,
                )
                for action_index in range(self._action_dim)
            ],
            dtype=np.float32,
        )


if __name__ == "__main__":
    env = gym.make("Hopper-v5")
    try:
        probe = ActionAxisProbe(100)
        probe.set_env(env)
        observation, _ = env.reset(seed=0)
        probe.reset(None)
        action = env.action_space.sample()
        probe.step(None, action, 0.0, None, False)
        observations, actions = probe.get_probes(observation)
        print(f"observations_shape={observations.shape}")
        print(f"actions_shape={actions.shape}")
    finally:
        env.close()
