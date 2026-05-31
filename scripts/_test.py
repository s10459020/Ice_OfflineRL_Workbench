from ice_offline.dataset.hopper_expert import HopperExpertDataset
from ice_offline.dataset.hopper_medium import HopperMediumDataset
from ice_offline.dataset.hopper_medium_d4rl import HopperMediumD4rlDataset
from ice_offline.dataset.hopper_medium_expert import HopperMediumExpertDataset
from ice_offline.dataset.hopper_medium_replay import HopperMediumReplayDataset
from ice_offline.dataset.hopper_simple import HopperSimpleDataset
from test_agent import test_aspl, test_bc, test_cql, test_iql, test_random, test_scas


EPISODES = 100


if __name__ == "__main__":
    # dataset = HopperSimpleDataset().load()
    # test_random.collect(dataset=dataset, task_id=f"{dataset.id}_random-v0", episodes=EPISODES)
    # test_bc.collect(dataset=dataset, task_id=f"{dataset.id}_bc-v0", episodes=EPISODES)
    # test_cql.collect(dataset=dataset, task_id=f"{dataset.id}_cql-v0", episodes=EPISODES)
    # test_iql.collect(dataset=dataset, task_id=f"{dataset.id}_iql-v0", episodes=EPISODES)
    # test_scas.collect(dataset=dataset, task_id=f"{dataset.id}_scas-v0", episodes=EPISODES)
    # test_aspl.collect(dataset=dataset, task_id=f"{dataset.id}_aspl-v0", episodes=EPISODES)

    # dataset = HopperMediumDataset().load()
    # test_random.collect(dataset=dataset, task_id=f"{dataset.id}_random-v0", episodes=EPISODES)
    # test_bc.collect(dataset=dataset, task_id=f"{dataset.id}_bc-v0", episodes=EPISODES)
    # test_cql.collect(dataset=dataset, task_id=f"{dataset.id}_cql-v0", episodes=EPISODES)
    # test_iql.collect(dataset=dataset, task_id=f"{dataset.id}_iql-v0", episodes=EPISODES)
    # test_scas.collect(dataset=dataset, task_id=f"{dataset.id}_scas-v0", episodes=EPISODES)
    # test_aspl.collect(dataset=dataset, task_id=f"{dataset.id}_aspl-v0", episodes=EPISODES)

    # dataset = HopperExpertDataset().load()
    # test_random.collect(dataset=dataset, task_id=f"{dataset.id}_random-v0", episodes=EPISODES)
    # test_bc.collect(dataset=dataset, task_id=f"{dataset.id}_bc-v0", episodes=EPISODES)
    # test_cql.collect(dataset=dataset, task_id=f"{dataset.id}_cql-v0", episodes=EPISODES)
    # test_iql.collect(dataset=dataset, task_id=f"{dataset.id}_iql-v0", episodes=EPISODES)
    # test_scas.collect(dataset=dataset, task_id=f"{dataset.id}_scas-v0", episodes=EPISODES)
    # test_aspl.collect(dataset=dataset, task_id=f"{dataset.id}_aspl-v0", episodes=EPISODES)

    # dataset = HopperMediumD4rlDataset().load()
    # test_random.collect(dataset=dataset, task_id=f"{dataset.id}_random-v0", episodes=EPISODES)
    # test_bc.collect(dataset=dataset, task_id=f"{dataset.id}_bc-v0", episodes=EPISODES)
    # test_cql.collect(dataset=dataset, task_id=f"{dataset.id}_cql-v0", episodes=EPISODES)
    # test_iql.collect(dataset=dataset, task_id=f"{dataset.id}_iql-v0", episodes=EPISODES)
    # test_scas.collect(dataset=dataset, task_id=f"{dataset.id}_scas-v0", episodes=EPISODES)
    # test_aspl.collect(dataset=dataset, task_id=f"{dataset.id}_aspl-v0", episodes=EPISODES)

    dataset = HopperMediumReplayDataset().load()
    test_random.collect(dataset=dataset, task_id=f"{dataset.id}_random-v0", episodes=EPISODES)
    test_bc.collect(dataset=dataset, task_id=f"{dataset.id}_bc-v0", episodes=EPISODES)
    test_cql.collect(dataset=dataset, task_id=f"{dataset.id}_cql-v0", episodes=EPISODES)
    test_iql.collect(dataset=dataset, task_id=f"{dataset.id}_iql-v0", episodes=EPISODES)
    test_scas.collect(dataset=dataset, task_id=f"{dataset.id}_scas-v0", episodes=EPISODES)
    test_aspl.collect(dataset=dataset, task_id=f"{dataset.id}_aspl-v0", episodes=EPISODES)

    dataset = HopperMediumExpertDataset().load()
    test_random.collect(dataset=dataset, task_id=f"{dataset.id}_random-v0", episodes=EPISODES)
    test_bc.collect(dataset=dataset, task_id=f"{dataset.id}_bc-v0", episodes=EPISODES)
    test_cql.collect(dataset=dataset, task_id=f"{dataset.id}_cql-v0", episodes=EPISODES)
    test_iql.collect(dataset=dataset, task_id=f"{dataset.id}_iql-v0", episodes=EPISODES)
    test_scas.collect(dataset=dataset, task_id=f"{dataset.id}_scas-v0", episodes=EPISODES)
    test_aspl.collect(dataset=dataset, task_id=f"{dataset.id}_aspl-v0", episodes=EPISODES)
