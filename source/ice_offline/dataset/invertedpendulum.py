import numpy as np

from ice_offline.dataset._spec import BaseDataset


class InvertedPendulumDataset(BaseDataset):
    dataset_name = "inverted_pendulum"
    dataset_id = "mujoco/invertedpendulum/expert-v0"
    env_id = "InvertedPendulum-v5"

    def obs_encode_batch(self, obs) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def obs_encode(self, obs) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def act_encode_batch(self, act) -> np.ndarray:
        return np.asarray(act, dtype=np.float32)

    def act_encode(self, act) -> np.ndarray:
        return np.asarray(act, dtype=np.float32)
