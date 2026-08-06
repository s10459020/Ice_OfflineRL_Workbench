from pathlib import Path

import h5py
import numpy as np
import torch


Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


DATASET_ROOT = Path(__file__).resolve().parent


# Available datasets for this compact build.
# Place the corresponding HDF5 files in source/joint_learning/datasets.
D4RL_DATASETS = {
    "hopper_medium": ("hopper_medium-v2.hdf5", "Hopper-v2"),
    "hopper_expert": ("hopper_expert-v2.hdf5", "Hopper-v2"),
    "hopper_hybrid": ("hopper_medium_expert-v2.hdf5", "Hopper-v2"),
    "hopper_medium_replay": ("hopper_medium_replay-v2.hdf5", "Hopper-v2"),
    "hopper_expert_replay": ("hopper_full_replay-v2.hdf5", "Hopper-v2"),
    "walker2d_medium": ("walker2d_medium-v2.hdf5", "Walker2d-v2"),
    "walker2d_expert": ("walker2d_expert-v2.hdf5", "Walker2d-v2"),
    "walker2d_hybrid": ("walker2d_medium_expert-v2.hdf5", "Walker2d-v2"),
    "walker2d_medium_replay": ("walker2d_medium_replay-v2.hdf5", "Walker2d-v2"),
    "walker2d_expert_replay": ("walker2d_full_replay-v2.hdf5", "Walker2d-v2"),
    "halfcheetah_medium": ("halfcheetah_medium-v2.hdf5", "HalfCheetah-v2"),
    "halfcheetah_expert": ("halfcheetah_expert-v2.hdf5", "HalfCheetah-v2"),
    "halfcheetah_hybrid": ("halfcheetah_medium_expert-v2.hdf5", "HalfCheetah-v2"),
    "halfcheetah_medium_replay": ("halfcheetah_medium_replay-v2.hdf5", "HalfCheetah-v2"),
    "halfcheetah_expert_replay": ("halfcheetah_full_replay-v2.hdf5", "HalfCheetah-v2"),
}


def load_dataset_info(dataset_id: str) -> tuple[str, int, int]:
    filename, env_id = D4RL_DATASETS[dataset_id]
    path = DATASET_ROOT / filename
    with h5py.File(path, "r") as h5_file:
        observations = h5_file["observations"]
        actions = h5_file["actions"]
        obs_size = int(np.prod(observations.shape[1:]))
        act_size = int(np.prod(actions.shape[1:]))
    return env_id, obs_size, act_size


class D4RLDataset:
    def __init__(self, dataset_id: str, device: str = "cuda") -> None:
        self.dataset_id = dataset_id
        self.device = device
        filename, env_id = D4RL_DATASETS[self.dataset_id]
        self.path = DATASET_ROOT / filename
        self.env_id = env_id
        self.load()

    @property
    def obs_size(self) -> int:
        return int(np.prod(self.observations.shape[1:]))

    @property
    def act_size(self) -> int:
        return int(np.prod(self.actions.shape[1:]))

    @property
    def count(self) -> int:
        return int(self.observations.shape[0])

    def load(self) -> None:
        with h5py.File(self.path, "r") as h5_file:
            terminals = np.asarray(h5_file["terminals"], dtype=np.bool_)
            timeouts = np.asarray(h5_file["timeouts"], dtype=np.bool_)
            dones = np.logical_or(terminals, timeouts).reshape(-1, 1)

            self.observations = torch.as_tensor(
                np.asarray(h5_file["observations"]),
                dtype=torch.float32,
                device=self.device,
            )
            self.actions = torch.as_tensor(
                np.asarray(h5_file["actions"]),
                dtype=torch.float32,
                device=self.device,
            )
            self.rewards = torch.as_tensor(
                np.asarray(h5_file["rewards"]).reshape(-1, 1),
                dtype=torch.float32,
                device=self.device,
            )
            self.next_observations = torch.as_tensor(
                np.asarray(h5_file["next_observations"]),
                dtype=torch.float32,
                device=self.device,
            )
            self.dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

    def sample_batch(self, batch_size: int) -> Batch:
        indexes = torch.randint(
            self.count,
            (batch_size,),
            device=self.device,
        )
        return (
            self.observations[indexes],
            self.actions[indexes],
            self.rewards[indexes],
            self.next_observations[indexes],
            self.dones[indexes],
        )
