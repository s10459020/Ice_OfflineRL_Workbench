import torch

from ice_offline.dataset._spec import BaseDataset
from ice_offline.run.evaluator import OnlineEvalFn


def eval_return(episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    _, _, reward, _, _ = episode_batch
    return {"return": float(reward.sum().item())}


DATASET_LOOKUP: dict[str, BaseDataset] = {
    "halfcheetah_expert": BaseDataset(dataset_name="halfcheetah", dataset_id="mujoco/halfcheetah/expert-v0", env_id="HalfCheetah-v5"),
    "halfcheetah_medium": BaseDataset(dataset_name="halfcheetah", dataset_id="mujoco/halfcheetah/medium-v0", env_id="HalfCheetah-v5"),
    "halfcheetah_simple": BaseDataset(dataset_name="halfcheetah", dataset_id="mujoco/halfcheetah/simple-v0", env_id="HalfCheetah-v5"),
    "hopper_expert": BaseDataset(dataset_name="hopper", dataset_id="mujoco/hopper/expert-v0", env_id="Hopper-v5"),
    "hopper_medium": BaseDataset(dataset_name="hopper", dataset_id="mujoco/hopper/medium-v0", env_id="Hopper-v5"),
    "hopper_simple": BaseDataset(dataset_name="hopper", dataset_id="mujoco/hopper/simple-v0", env_id="Hopper-v5"),
    "invertedpendulum_expert": BaseDataset(dataset_name="inverted_pendulum", dataset_id="mujoco/invertedpendulum/expert-v0", env_id="InvertedPendulum-v5"),
    "onerooms8_fullobs_optimal": BaseDataset(dataset_name="onerooms8", dataset_id="minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0", env_id="BabyAI-OneRoomS8-v0"),
    "walker2d_expert": BaseDataset(dataset_name="walker2d", dataset_id="mujoco/walker2d/expert-v0", env_id="Walker2d-v5"),
    "walker2d_medium": BaseDataset(dataset_name="walker2d", dataset_id="mujoco/walker2d/medium-v0", env_id="Walker2d-v5"),
    "walker2d_simple": BaseDataset(dataset_name="walker2d", dataset_id="mujoco/walker2d/simple-v0", env_id="Walker2d-v5"),
}

DATASET_EVAL_ONLINE_LOOKUP: dict[str, list[OnlineEvalFn]] = {
    "halfcheetah_expert": [eval_return],
    "halfcheetah_medium": [eval_return],
    "halfcheetah_simple": [eval_return],
    "hopper_expert": [eval_return],
    "hopper_medium": [eval_return],
    "hopper_simple": [eval_return],
    "invertedpendulum_expert": [eval_return],
    "onerooms8_fullobs_optimal": [eval_return],
    "walker2d_expert": [eval_return],
    "walker2d_medium": [eval_return],
    "walker2d_simple": [eval_return],
}


def get_dataset(dataset_id: str) -> BaseDataset:
    if dataset_id == "onerooms8_fullobs_optimal":
        from ice_offline.dataset.oneroom_s8 import OneRoomS8Dataset

        return OneRoomS8Dataset()
    return DATASET_LOOKUP[dataset_id]


def get_dataset_train_bundle(dataset_id: str) -> tuple[BaseDataset, list[OnlineEvalFn]]:
    return (
        get_dataset(dataset_id),
        DATASET_EVAL_ONLINE_LOOKUP[dataset_id],
    )
