from ice_offline.dataset._spec import BaseDataset
from ice_offline.pipeline.d4rl.loader import D4rlLoader
from ice_offline.pipeline.minari.loader import MinariLoader


def _build_dataset(dataset_key: str) -> BaseDataset:
    if dataset_key == "halfcheetah_expert":
        return BaseDataset("HalfCheetah-v5", MinariLoader("mujoco/halfcheetah/expert-v0").buffer)
    if dataset_key == "halfcheetah_medium":
        return BaseDataset("HalfCheetah-v5", MinariLoader("mujoco/halfcheetah/medium-v0").buffer)
    if dataset_key == "halfcheetah_simple":
        return BaseDataset("HalfCheetah-v5", MinariLoader("mujoco/halfcheetah/simple-v0").buffer)
    if dataset_key == "hopper_expert":
        return BaseDataset("Hopper-v5", MinariLoader("mujoco/hopper/expert-v0").buffer)
    if dataset_key == "hopper_medium":
        return BaseDataset("Hopper-v5", MinariLoader("mujoco/hopper/medium-v0").buffer)
    if dataset_key == "hopper_simple":
        return BaseDataset("Hopper-v5", MinariLoader("mujoco/hopper/simple-v0").buffer)
    if dataset_key == "hopper_medium_d4rl":
        return BaseDataset("Hopper-v5", D4rlLoader("tmps/datasets/d4rl/hopper_medium-v2.hdf5").buffer)
    if dataset_key == "hopper_medium_replay_d4rl":
        return BaseDataset("Hopper-v5", D4rlLoader("tmps/datasets/d4rl/hopper_medium_replay-v2.hdf5").buffer)
    if dataset_key == "hopper_medium_expert_d4rl":
        return BaseDataset("Hopper-v5", D4rlLoader("tmps/datasets/d4rl/hopper_medium_expert-v2.hdf5").buffer)
    if dataset_key == "invertedpendulum_expert":
        return BaseDataset("InvertedPendulum-v5", MinariLoader("mujoco/invertedpendulum/expert-v0").buffer)
    if dataset_key == "walker2d_expert":
        return BaseDataset("Walker2d-v5", MinariLoader("mujoco/walker2d/expert-v0").buffer)
    if dataset_key == "walker2d_medium":
        return BaseDataset("Walker2d-v5", MinariLoader("mujoco/walker2d/medium-v0").buffer)
    if dataset_key == "walker2d_simple":
        return BaseDataset("Walker2d-v5", MinariLoader("mujoco/walker2d/simple-v0").buffer)
    if dataset_key == "onerooms8_fullobs_optimal":
        from ice_offline.dataset.oneroom_s8 import OneRoomS8Dataset
        return OneRoomS8Dataset(raw_buffer=MinariLoader("minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0").buffer)
    raise KeyError(dataset_key)


def get_dataset(dataset_key: str) -> BaseDataset:
    return _build_dataset(dataset_key)
