import torch

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
from train_agent import train_aspl
from train_agent import train_bc_deterministic
from train_agent import train_bc_stochastic
from train_agent import train_cql
from train_agent import train_cql_max_q
from train_agent import train_cql_soft_q
from train_agent import train_iql
from train_agent import train_sdc_cql
from train_agent import train_sdc_cql_pre
from train_agent import train_scas_aspl
from train_agent import train_scas_mean
from train_agent import train_scas_min
from train_agent import train_td3bc


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

TRAIN_KWARGS = {
    "steps": 200_000,
    "save_interval": 20_000,
    "eval_interval": 2_000,
    "eval_online_n": 20,
    "eval_offline_n": 30,
}

SCAS_KWARGS = {
    "steps_dynamic": 100_000,
}

SDC_PRE_KWARGS = {
    "steps_model": 100_000,
}

DATASET_LIST = [
    # HopperRandomDataset,
    # HopperReplayDataset,
    # HopperMediumReplayDataset,
    # HopperMediumD4rlDataset,
    # HopperMediumExpertDataset,
    # HopperExpertD4rlDataset,
    # HopperSimpleDataset,
    # HopperMediumDataset,
    # HopperExpertDataset,
    # HalfCheetahRandomDataset,
    # HalfCheetahReplayDataset,
    # HalfCheetahMediumReplayDataset,
    # HalfCheetahMediumD4rlDataset,
    # HalfCheetahMediumExpertDataset,
    # HalfCheetahExpertD4rlDataset,
    # HalfCheetahSimpleDataset,
    # HalfCheetahMediumDataset,
    # HalfCheetahExpertDataset,
    Walker2dRandomDataset,
    Walker2dReplayDataset,
    Walker2dMediumReplayDataset,
    Walker2dMediumD4rlDataset,
    Walker2dMediumExpertDataset,
    Walker2dExpertD4rlDataset,
    Walker2dSimpleDataset,
    Walker2dMediumDataset,
    Walker2dExpertDataset,
]

AGENT_LIST = [
    ("bc_deterministic", train_bc_deterministic.collect),
    ("bc_stochastic", train_bc_stochastic.collect),
    ("td3bc", train_td3bc.collect),
    ("iql", train_iql.collect),
    ("cql", train_cql.collect),
    ("sdc_cql", train_sdc_cql.collect),
    ("sdc_cql_pre", train_sdc_cql_pre.collect),
    ("cql_max_q", train_cql_max_q.collect),
    ("cql_soft_q", train_cql_soft_q.collect),
    ("aspl", train_aspl.collect),
    ("scas_aspl", train_scas_aspl.collect),
    ("scas_mean", train_scas_mean.collect),
    ("scas_min", train_scas_min.collect),
]


def collect_agent(agent_id: str, trainer, dataset):
    train_kwargs = {k: v for k, v in TRAIN_KWARGS.items() if v is not None}
    if agent_id in ("scas_aspl", "scas_mean", "scas_min"):
        train_kwargs.update({k: v for k, v in SCAS_KWARGS.items() if v is not None})
    if agent_id == "sdc_cql_pre":
        train_kwargs.update({k: v for k, v in SDC_PRE_KWARGS.items() if v is not None})

    task_id = f"{dataset.id}-{agent_id}-v0"
    return trainer(dataset=dataset, task_id=task_id, device=DEVICE, **train_kwargs)


if __name__ == "__main__":
    for dataset_cls in DATASET_LIST:
        dataset = dataset_cls(device=DEVICE).load()

        for agent_id, trainer in AGENT_LIST:
            print(f"dataset={dataset.id}, agent={agent_id}")
            minari_data, state_data = collect_agent(agent_id, trainer, dataset)
            print(f"dataset_id={minari_data.spec.dataset_id}")
            print(f"total_episodes={minari_data.total_episodes}")
            print(f"total_steps={minari_data.total_steps}")
