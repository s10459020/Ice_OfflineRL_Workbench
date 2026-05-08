from ice_offline.dataset._spec import BaseDataset


class InvertedPendulumDataset(BaseDataset):
    dataset_name = "inverted_pendulum"
    dataset_id = "mujoco/invertedpendulum/expert-v0"
    env_id = "InvertedPendulum-v5"
