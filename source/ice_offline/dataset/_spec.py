import gymnasium as gym
import numpy as np


class BaseDataset:
    env_id: str

    def __init__(
        self,
        env_id: str,
        buffer,
    ) -> None:
        self.env_id = env_id
        self.raw_buffer = buffer
        self.buffer = self._build(self.raw_buffer)

        self._rng = np.random.default_rng()
        self.obs_shape = tuple(int(x) for x in self.buffer["obs"].shape[1:])
        self.act_shape = tuple(int(x) for x in self.buffer["act"].shape[1:])
        self.obs_dim = int(np.prod(self.obs_shape)) if self.obs_shape else 1
        self.act_dim = int(np.prod(self.act_shape)) if self.act_shape else 1
        self.count = int(self.buffer["obs"].shape[0])

    def set_seed(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def obs_encode_batch(self, obs):
        return np.asarray(obs, dtype=np.float32)

    def act_encode_batch(self, act):
        return np.asarray(act, dtype=np.float32)

    def sample_batch(self, batch_size: int):
        idx = self._rng.integers(0, self.count, size=(batch_size,))
        return {
            "obs": self.buffer["obs"][idx],
            "next_obs": self.buffer["next_obs"][idx],
            "act": self.buffer["act"][idx],
            "rew": self.buffer["rew"][idx],
            "done": self.buffer["done"][idx],
        }

    def _build(self, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        obs = self.obs_encode_batch(np.asarray(raw["observations"], dtype=np.float32))
        next_obs = self.obs_encode_batch(np.asarray(raw["next_observations"], dtype=np.float32))
        act = self.act_encode_batch(np.asarray(raw["actions"], dtype=np.float32))
        rew = np.asarray(raw["rewards"], dtype=np.float32)
        term = np.asarray(raw["terminations"], dtype=np.bool_)
        trunc = np.asarray(raw["truncations"], dtype=np.bool_)
        done = np.logical_or(term, trunc).astype(np.float32)
        return {
            "obs": obs,
            "next_obs": next_obs,
            "act": act,
            "rew": rew,
            "done": done,
        }

    def make_env(self):
        return gym.make(self.env_id)

    def make_collect_env(self):
        return gym.make(self.env_id)

    def make_render_env(self):
        return gym.make(self.env_id, render_mode="human")
