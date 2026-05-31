from ice_offline.dataset.hopper_medium_expert import HopperMediumExpertDataset
from ice_offline.dataset.hopper_medium_replay import HopperMediumReplayDataset
from ice_offline.run import test_aspl, test_bc, test_cql, test_iql, test_random, test_scas


EPISODES = 100
DATASET_CLASSES = [
    HopperMediumReplayDataset,
    HopperMediumExpertDataset,
]


def collect_all(dataset) -> None:
    test_random.collect(dataset, task_id=f"{dataset.id}_random-v0", episodes=EPISODES)
    test_bc.collect(dataset, task_id=f"{dataset.id}_bc-v0", episodes=EPISODES)
    test_cql.collect(dataset, task_id=f"{dataset.id}_cql-v0", episodes=EPISODES)
    test_iql.collect(dataset, task_id=f"{dataset.id}_iql-v0", episodes=EPISODES)
    test_scas.collect(dataset, task_id=f"{dataset.id}_scas-v0", episodes=EPISODES)
    test_aspl.collect(dataset, task_id=f"{dataset.id}_aspl-v0", episodes=EPISODES)


if __name__ == "__main__":
    for dataset_class in DATASET_CLASSES:
        collect_all(dataset_class().load())
