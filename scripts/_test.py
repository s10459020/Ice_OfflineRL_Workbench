from ice_offline.dataset.halfcheetah_expert import HalfCheetahExpertDataset
from ice_offline.dataset.halfcheetah_expert_d4rl import HalfCheetahExpertD4rlDataset
from ice_offline.dataset.halfcheetah_medium import HalfCheetahMediumDataset
from ice_offline.dataset.halfcheetah_medium_d4rl import HalfCheetahMediumD4rlDataset
from ice_offline.dataset.halfcheetah_medium_expert import HalfCheetahMediumExpertDataset
from ice_offline.dataset.halfcheetah_medium_replay import HalfCheetahMediumReplayDataset
from ice_offline.dataset.halfcheetah_random import HalfCheetahRandomDataset
from ice_offline.dataset.halfcheetah_replay import HalfCheetahReplayDataset
from ice_offline.dataset.halfcheetah_simple import HalfCheetahSimpleDataset
from ice_offline.dataset.hopper_expert import HopperExpertDataset
from ice_offline.dataset.hopper_expert_d4rl import HopperExpertD4rlDataset
from ice_offline.dataset.hopper_medium import HopperMediumDataset
from ice_offline.dataset.hopper_medium_d4rl import HopperMediumD4rlDataset
from ice_offline.dataset.hopper_medium_expert import HopperMediumExpertDataset
from ice_offline.dataset.hopper_medium_replay import HopperMediumReplayDataset
from ice_offline.dataset.hopper_random import HopperRandomDataset
from ice_offline.dataset.hopper_replay import HopperReplayDataset
from ice_offline.dataset.hopper_simple import HopperSimpleDataset
from test_agent import test_aspl
from test_agent import test_bc_deterministic
from test_agent import test_bc_stochastic
from test_agent import test_cql
from test_agent import test_cql_max_q
from test_agent import test_cql_soft_q
from test_agent import test_iql
from test_agent import test_random
from test_agent import test_sdc_cql
from test_agent import test_scas_mean
from test_agent import test_scas_min
from test_agent import test_td3bc


EPISODES = 100
PRINT_INTERVAL = 1

DATASET_LIST = [
    # HopperRandomDataset,
    # HopperReplayDataset,
    HopperMediumReplayDataset,
    HopperMediumD4rlDataset,
    HopperMediumExpertDataset,
    # HopperExpertD4rlDataset,
    HopperSimpleDataset,
    HopperMediumDataset,
    HopperExpertDataset,
    # HalfCheetahRandomDataset,
    # HalfCheetahReplayDataset,
    # HalfCheetahMediumReplayDataset,
    # HalfCheetahMediumD4rlDataset,
    # HalfCheetahMediumExpertDataset,
    # HalfCheetahExpertD4rlDataset,
    # HalfCheetahSimpleDataset,
    # HalfCheetahMediumDataset,
    # HalfCheetahExpertDataset,
]

AGENT_LIST = [
    ("bc_deterministic", test_bc_deterministic.collect),
    ("bc_stochastic", test_bc_stochastic.collect),
    ("td3bc", test_td3bc.collect),
    ("iql", test_iql.collect),
    ("cql", test_cql.collect),
    ("sdc_cql", test_sdc_cql.collect),
    ("cql_max_q", test_cql_max_q.collect),
    ("cql_soft_q", test_cql_soft_q.collect),
    ("aspl", test_aspl.collect),
    ("scas_mean", test_scas_mean.collect),
    ("scas_min", test_scas_min.collect),
    ("random", test_random.collect),
]

TEST_KWARGS = {
    "episodes": EPISODES,
    "print_interval": PRINT_INTERVAL,
}


def collect_agent(agent_id: str, tester, dataset):
    test_kwargs = {k: v for k, v in TEST_KWARGS.items() if v is not None}
    task_id = f"{dataset.id}-{agent_id}-v0"
    return tester(dataset=dataset, task_id=task_id, **test_kwargs)


if __name__ == "__main__":
    for dataset_cls in DATASET_LIST:
        dataset = dataset_cls().load()

        for agent_id, tester in AGENT_LIST:
            print(f"dataset={dataset.id}, agent={agent_id}")
            returns, minari_data, state_data = collect_agent(agent_id, tester, dataset)
            print(f"avg_returns={sum(returns) / len(returns):.2f}")
            print(f"dataset_id={minari_data.spec.dataset_id}")
            print(f"total_episodes={minari_data.total_episodes}")
            print(f"total_steps={minari_data.total_steps}")
