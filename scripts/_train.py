import torch

from ice_offline.dataset.hopper_expert import HopperExpertDataset
from ice_offline.dataset.hopper_medium import HopperMediumDataset
from ice_offline.dataset.hopper_medium_d4rl import HopperMediumD4rlDataset
from ice_offline.dataset.hopper_medium_expert import HopperMediumExpertDataset
from ice_offline.dataset.hopper_medium_replay import HopperMediumReplayDataset
from ice_offline.dataset.hopper_simple import HopperSimpleDataset
from train_agent import train_aspl
from train_agent import train_bc_deterministic
from train_agent import train_bc_stochastic
from train_agent import train_cql
from train_agent import train_cql_max_q
from train_agent import train_cql_soft_q
from train_agent import train_iql
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

DATASET_LIST = [
    HopperSimpleDataset,
    HopperMediumDataset,
    HopperExpertDataset,
    HopperMediumD4rlDataset,
    HopperMediumReplayDataset,
    HopperMediumExpertDataset,
]

AGENT_LIST = [
    # ("bc_deterministic", train_bc_deterministic.collect),
    # ("bc_stochastic", train_bc_stochastic.collect),
    # ("td3bc", train_td3bc.collect),
    # ("iql", train_iql.collect),
    # ("cql", train_cql.collect),
     ("cql_max_q", train_cql_max_q.collect),
     ("cql_soft_q", train_cql_soft_q.collect),
     ("aspl", train_aspl.collect),
     ("scas_mean", train_scas_mean.collect),
     ("scas_min", train_scas_min.collect),
]


def collect_agent(agent_id: str, trainer, dataset):
    train_kwargs = {k: v for k, v in TRAIN_KWARGS.items() if v is not None}
    if agent_id in ("scas_mean", "scas_min"):
        train_kwargs.update({k: v for k, v in SCAS_KWARGS.items() if v is not None})

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
