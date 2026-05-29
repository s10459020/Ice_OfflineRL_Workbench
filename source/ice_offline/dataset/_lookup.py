from pathlib import Path

from ice_offline.dataset._spec import BaseDataset
from ice_offline.pipeline.d4rl.loader import D4rlLoader
from ice_offline.pipeline.minari.loader import MinariLoader


DATASET_SPECS: dict[str, dict[str, str]] = {
    "halfcheetah_expert": {"type": "minari", "dataset_path": "tmps/datasets/mujoco/halfcheetah/expert-v0/data/main_data.hdf5", "env_id": "HalfCheetah-v5"},
    "halfcheetah_medium": {"type": "minari", "dataset_path": "tmps/datasets/mujoco/halfcheetah/medium-v0/data/main_data.hdf5", "env_id": "HalfCheetah-v5"},
    "halfcheetah_simple": {"type": "minari", "dataset_path": "tmps/datasets/mujoco/halfcheetah/simple-v0/data/main_data.hdf5", "env_id": "HalfCheetah-v5"},
    "hopper_expert": {"type": "minari", "dataset_path": "tmps/datasets/mujoco/hopper/expert-v0/data/main_data.hdf5", "env_id": "Hopper-v5"},
    "hopper_medium": {"type": "minari", "dataset_path": "tmps/datasets/mujoco/hopper/medium-v0/data/main_data.hdf5", "env_id": "Hopper-v5"},
    "hopper_simple": {"type": "minari", "dataset_path": "tmps/datasets/mujoco/hopper/simple-v0/data/main_data.hdf5", "env_id": "Hopper-v5"},
    "hopper_medium_d4rl": {"type": "d4rl", "dataset_path": "tmps/datasets/d4rl/hopper_medium-v2.hdf5", "env_id": "Hopper-v5"},
    "hopper_medium_replay_d4rl": {"type": "d4rl", "dataset_path": "tmps/datasets/d4rl/hopper_medium_replay-v2.hdf5", "env_id": "Hopper-v5"},
    "hopper_medium_expert_d4rl": {"type": "d4rl", "dataset_path": "tmps/datasets/d4rl/hopper_medium_expert-v2.hdf5", "env_id": "Hopper-v5"},
    "invertedpendulum_expert": {"type": "minari", "dataset_path": "tmps/datasets/mujoco/invertedpendulum/expert-v0/data/main_data.hdf5", "env_id": "InvertedPendulum-v5"},
    "walker2d_expert": {"type": "minari", "dataset_path": "tmps/datasets/mujoco/walker2d/expert-v0/data/main_data.hdf5", "env_id": "Walker2d-v5"},
    "walker2d_medium": {"type": "minari", "dataset_path": "tmps/datasets/mujoco/walker2d/medium-v0/data/main_data.hdf5", "env_id": "Walker2d-v5"},
    "walker2d_simple": {"type": "minari", "dataset_path": "tmps/datasets/mujoco/walker2d/simple-v0/data/main_data.hdf5", "env_id": "Walker2d-v5"},
    "onerooms8_fullobs_optimal": {"type": "onerooms8", "dataset_path": "tmps/datasets/minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0/data/main_data.hdf5", "env_id": "BabyAI-OneRoomS8-v0"},
}


def get_env_id(dataset_key: str) -> str:
    return DATASET_SPECS[dataset_key]["env_id"]


def _build_dataset(dataset_key: str) -> BaseDataset:
    spec = DATASET_SPECS[dataset_key]
    dataset_type = spec["type"]
    dataset_path = spec["dataset_path"]
    env_id = spec["env_id"]

    if dataset_type == "minari":
        return BaseDataset(env_id, MinariLoader(dataset_path).buffer)

    if dataset_type == "d4rl":
        return BaseDataset(env_id, D4rlLoader(dataset_path).buffer)

    if dataset_type == "onerooms8":
        from ice_offline.dataset.oneroom_s8 import OneRoomS8Dataset

        return OneRoomS8Dataset(raw_buffer=MinariLoader(dataset_path).buffer)

    raise KeyError(dataset_key)


def get_dataset(dataset_key: str) -> BaseDataset:
    return _build_dataset(dataset_key)
