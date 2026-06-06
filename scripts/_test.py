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
from ice_offline.dataset.walker2d_expert import Walker2dExpertDataset
from ice_offline.dataset.walker2d_expert_d4rl import Walker2dExpertD4rlDataset
from ice_offline.dataset.walker2d_medium import Walker2dMediumDataset
from ice_offline.dataset.walker2d_medium_d4rl import Walker2dMediumD4rlDataset
from ice_offline.dataset.walker2d_medium_expert import Walker2dMediumExpertDataset
from ice_offline.dataset.walker2d_medium_replay import Walker2dMediumReplayDataset
from ice_offline.dataset.walker2d_random import Walker2dRandomDataset
from ice_offline.dataset.walker2d_replay import Walker2dReplayDataset
from ice_offline.dataset.walker2d_simple import Walker2dSimpleDataset
from test_agent import test_aspl
from test_agent import test_bc_deterministic
from test_agent import test_bc_stochastic
from test_agent import test_cql
from test_agent import test_cql_max_q
from test_agent import test_cql_soft_q
from test_agent import test_iql
from test_agent import test_random
from test_agent import test_sdc_cql
from test_agent import test_sdc_pre
from test_agent import test_scas_aspl
from test_agent import test_scas_mean
from test_agent import test_scas_min
from test_agent import test_td3bc

TEST_KWARGS = {
    "episodes": 100,
    "print_interval": 1,
}

SDC_KWARGS = {
    "model_step": 100_000,
}

SCAS_KWARGS = {
    "model_step": 100_000,
    "dynamic_step": 100_000,
}

DATASET_LIST = [
    HopperRandomDataset,
    HopperReplayDataset,
    HopperMediumReplayDataset,
    HopperMediumD4rlDataset,
    HopperMediumExpertDataset,
    HopperExpertD4rlDataset,
    HopperSimpleDataset,
    HopperMediumDataset,
    HopperExpertDataset,
    Walker2dRandomDataset,
    Walker2dReplayDataset,
    Walker2dMediumReplayDataset,
    Walker2dMediumD4rlDataset,
    Walker2dMediumExpertDataset,
    Walker2dExpertD4rlDataset,
    Walker2dSimpleDataset,
    Walker2dMediumDataset,
    Walker2dExpertDataset,
    HalfCheetahRandomDataset,
    HalfCheetahReplayDataset,
    HalfCheetahMediumReplayDataset,
    HalfCheetahMediumD4rlDataset,
    HalfCheetahMediumExpertDataset,
    HalfCheetahExpertD4rlDataset,
    HalfCheetahSimpleDataset,
    HalfCheetahMediumDataset,
    HalfCheetahExpertDataset,
]

AGENT_LIST = [
    ("random", test_random.collect),
    ("bc_deterministic", test_bc_deterministic.collect),
    ("bc_stochastic", test_bc_stochastic.collect),
    ("td3bc", test_td3bc.collect),
    ("iql", test_iql.collect),
    ("cql", test_cql.collect),
    ("cql_max_q", test_cql_max_q.collect),
    ("cql_soft_q", test_cql_soft_q.collect),
    ("aspl", test_aspl.collect),
    ("sdc_cql", test_sdc_cql.collect),
    ("sdc_pre", test_sdc_pre.collect),
    ("scas_min", test_scas_min.collect),
    ("scas_mean", test_scas_mean.collect),
    ("scas_aspl", test_scas_aspl.collect),
]


def collect_agent(agent_id: str, tester, dataset):
    test_kwargs = {k: v for k, v in TEST_KWARGS.items() if v is not None}
    if agent_id in ("scas_aspl", "scas_mean", "scas_min"):
        test_kwargs.update({k: v for k, v in SCAS_KWARGS.items() if v is not None})
    elif agent_id != "random":
        test_kwargs.update({k: v for k, v in SDC_KWARGS.items() if v is not None})
    return tester(dataset=dataset, **test_kwargs)


if __name__ == "__main__":
    for dataset_cls in DATASET_LIST:
        dataset = dataset_cls()

        for agent_id, tester in AGENT_LIST:
            print(f"dataset={dataset.id}, agent={agent_id}")
            returns, minari_data, state_data = collect_agent(agent_id, tester, dataset)
            print(f"avg_returns={sum(returns) / len(returns):.2f}")
            print(f"dataset_id={minari_data.spec.dataset_id}")
            print(f"total_episodes={minari_data.total_episodes}")
            print(f"total_steps={minari_data.total_steps}")
