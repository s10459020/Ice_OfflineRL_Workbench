import copy

from ice_offline.dataset._spec import BaseDataset
from ice_offline.dataset.oneroom_s8 import OneRoomS8Dataset
from ice_offline.dataset._spec import eval_return
from ice_offline.dataset._spec import StopReturnStable
from ice_offline.runner.evaluator import OnlineEvalFn
from ice_offline.runner.stopper import EarlyStopEvent


DATASET_LOOKUP: dict[str, BaseDataset] = {
    "halfcheetah_expert": BaseDataset(dataset_name="halfcheetah", dataset_id="mujoco/halfcheetah/expert-v0", env_id="HalfCheetah-v5"),
    "halfcheetah_medium": BaseDataset(dataset_name="halfcheetah", dataset_id="mujoco/halfcheetah/medium-v0", env_id="HalfCheetah-v5"),
    "halfcheetah_simple": BaseDataset(dataset_name="halfcheetah", dataset_id="mujoco/halfcheetah/simple-v0", env_id="HalfCheetah-v5"),
    "hopper_expert": BaseDataset(dataset_name="hopper", dataset_id="mujoco/hopper/expert-v0", env_id="Hopper-v5"),
    "hopper_medium": BaseDataset(dataset_name="hopper", dataset_id="mujoco/hopper/medium-v0", env_id="Hopper-v5"),
    "hopper_simple": BaseDataset(dataset_name="hopper", dataset_id="mujoco/hopper/simple-v0", env_id="Hopper-v5"),
    "invertedpendulum_expert": BaseDataset(dataset_name="inverted_pendulum", dataset_id="mujoco/invertedpendulum/expert-v0", env_id="InvertedPendulum-v5"),
    "onerooms8_fullobs_optimal": OneRoomS8Dataset(),
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

DATASET_STOP_LOOKUP: dict[str, list[EarlyStopEvent]] = {
    "halfcheetah_expert": [StopReturnStable(patience=5, lambda_ratio=0.01)],
    "halfcheetah_medium": [StopReturnStable(patience=5, lambda_ratio=0.01)],
    "halfcheetah_simple": [StopReturnStable(patience=5, lambda_ratio=0.01)],
    "hopper_expert": [StopReturnStable(patience=5, lambda_ratio=0.01)],
    "hopper_medium": [StopReturnStable(patience=5, lambda_ratio=0.01)],
    "hopper_simple": [StopReturnStable(patience=5, lambda_ratio=0.01)],
    "invertedpendulum_expert": [StopReturnStable(patience=5, lambda_ratio=0.01)],
    "onerooms8_fullobs_optimal": [StopReturnStable(patience=5, lambda_ratio=0.01)],
    "walker2d_expert": [StopReturnStable(patience=5, lambda_ratio=0.01)],
    "walker2d_medium": [StopReturnStable(patience=5, lambda_ratio=0.01)],
    "walker2d_simple": [StopReturnStable(patience=5, lambda_ratio=0.01)],
}


def get_dataset(dataset_id: str) -> BaseDataset:
    return DATASET_LOOKUP[dataset_id]


def get_dataset_train_bundle(dataset_id: str) -> tuple[BaseDataset, list[OnlineEvalFn], list[EarlyStopEvent]]:
    early_stop_events = copy.deepcopy(DATASET_STOP_LOOKUP[dataset_id])
    return (
        get_dataset(dataset_id),
        DATASET_EVAL_ONLINE_LOOKUP[dataset_id],
        early_stop_events,
    )

