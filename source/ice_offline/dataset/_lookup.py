import copy

from ice_offline.dataset import BaseDataset
from ice_offline.dataset import InvertedPendulumDataset
from ice_offline.dataset import OneRoomS8Dataset
from ice_offline.dataset._spec import eval_return
from ice_offline.dataset._spec import StopReturnStable
from ice_offline.runner.offline import EarlyStopEvent
from ice_offline.runner.offline import OnlineEvalFn

DATASET_LOOKUP: dict[str, BaseDataset] = {
    "invertedpendulum_expert": InvertedPendulumDataset(),
    "invertedpendulum_expert_early": InvertedPendulumDataset(),
    "onerooms8_fullobs_optimal": OneRoomS8Dataset(),
    "onerooms8_fullobs_optimal_early": OneRoomS8Dataset(),
}

DATASET_EVAL_ONLINE_LOOKUP: dict[str, list[OnlineEvalFn]] = {
    "invertedpendulum_expert": [eval_return],
    "invertedpendulum_expert_early": [eval_return],
    "onerooms8_fullobs_optimal": [eval_return],
    "onerooms8_fullobs_optimal_early": [eval_return],
}

DATASET_EARLY_STOP_LOOKUP: dict[str, list[EarlyStopEvent]] = {
    "invertedpendulum_expert": [],
    "invertedpendulum_expert_early": [StopReturnStable(patience=5, lambda_ratio=0.01)],
    "onerooms8_fullobs_optimal": [],
    "onerooms8_fullobs_optimal_early": [StopReturnStable(patience=5, lambda_ratio=0.01)],
}


def get_dataset(dataset_id: str) -> BaseDataset:
    return DATASET_LOOKUP[dataset_id]


def get_dataset_train_bundle(dataset_id: str) -> tuple[BaseDataset, list[OnlineEvalFn], list[EarlyStopEvent]]:
    early_stop_events = copy.deepcopy(DATASET_EARLY_STOP_LOOKUP[dataset_id])
    return (
        get_dataset(dataset_id),
        DATASET_EVAL_ONLINE_LOOKUP[dataset_id],
        early_stop_events,
    )
