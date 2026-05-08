from ice_offline.dataset import BaseDataset
from ice_offline.dataset import InvertedPendulumDataset
from ice_offline.dataset import OneRoomS8Dataset
from ice_offline.dataset._spec import eval_return
from ice_offline.dataset._spec import eval_reward_sum
from ice_offline.runner.offline import OnlineEvalFn

DATASET_ONEROOMS8_FULLOBS_OPTIMAL = "onerooms8_fullobs_optimal"
DATASET_INVERTEDPENDULUM_EXPERT = "invertedpendulum_expert"

DATASET_LOOKUP: dict[str, BaseDataset] = {
    DATASET_ONEROOMS8_FULLOBS_OPTIMAL: OneRoomS8Dataset(),
    DATASET_INVERTEDPENDULUM_EXPERT: InvertedPendulumDataset(),
}

DATASET_EVAL_ONLINE_LOOKUP: dict[str, list[OnlineEvalFn]] = {
    DATASET_ONEROOMS8_FULLOBS_OPTIMAL: [eval_return],
    DATASET_INVERTEDPENDULUM_EXPERT: [eval_reward_sum],
}


def get_dataset(dataset_id: str) -> BaseDataset:
    return DATASET_LOOKUP[dataset_id]


def get_dataset_train_bundle(dataset_id: str) -> tuple[BaseDataset, list[OnlineEvalFn]]:
    dataset = get_dataset(dataset_id)
    eval_online_fns = DATASET_EVAL_ONLINE_LOOKUP[dataset_id]
    return dataset, eval_online_fns
