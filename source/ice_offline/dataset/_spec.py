import gymnasium as gym
import minari
import numpy as np
from pathlib import Path

from ice_offline.tools.paths import minari_root


def dataset_ref(dataset_id: str) -> Path:
    return minari_root() / dataset_id / "data"


class BaseDataset:
    dataset_name: str
    dataset_id: str
    env_id: str

    def __init__(
        self,
        dataset_name: str,
        dataset_id: str,
        env_id: str,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self.env_id = env_id

    def obs_encode_batch(self, obs):
        return np.asarray(obs, dtype=np.float32)

    def obs_encode(self, obs):
        return np.asarray(obs, dtype=np.float32)

    def act_encode_batch(self, act):
        return np.asarray(act, dtype=np.float32)

    def act_encode(self, act):
        return np.asarray(act, dtype=np.float32)

    def make_dataset(self):
        return minari.load_dataset(self.dataset_id, download=True)

    def make_collect_env(self):
        return gym.make(self.env_id)

    def make_render_env(self):
        return gym.make(self.env_id, render_mode="human")

    def observation_cardinality(self, observation_shape: tuple[int, ...], minari_dataset) -> tuple[int, ...] | None:
        return None

    def action_cardinality(self, action_shape: tuple[int, ...], minari_dataset) -> tuple[int, ...] | None:
        if action_shape != (1,):
            return None
        n = getattr(minari_dataset.spec.action_space, "n", None)
        if n is None:
            return None
        return (int(n),)
