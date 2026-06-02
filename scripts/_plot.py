from pathlib import Path

import matplotlib

from ice_offline.dataset.hopper_expert import HopperExpertDataset
from ice_offline.dataset.hopper_expert_d4rl import HopperExpertD4rlDataset
from ice_offline.dataset.hopper_medium import HopperMediumDataset
from ice_offline.dataset.hopper_medium_d4rl import HopperMediumD4rlDataset
from ice_offline.dataset.hopper_medium_expert import HopperMediumExpertDataset
from ice_offline.dataset.hopper_medium_replay import HopperMediumReplayDataset
from ice_offline.dataset.hopper_random import HopperRandomDataset
from ice_offline.dataset.hopper_replay import HopperReplayDataset
from ice_offline.dataset.hopper_simple import HopperSimpleDataset

matplotlib.use("Agg")

from ice_offline.plot.plotter import plot_csv


EVAL_ROOT = Path("tmps/eval")
PLOT_ROOT = Path("plot/agent")
SHOW = False

DATASET_LIST = [
    HopperRandomDataset,
    HopperReplayDataset,
    HopperExpertD4rlDataset,
    HopperMediumD4rlDataset,
    HopperMediumReplayDataset,
    HopperMediumExpertDataset,
    HopperSimpleDataset,
    HopperMediumDataset,
    HopperExpertDataset,
]

AGENT_LIST = [
    "random",
    "bc_deterministic",
    "bc_stochastic",
    "td3bc",
    "iql",
    "cql",
    "cql_max_q",
    "cql_soft_q",
    "aspl",
    "scas_mean",
    "scas_min",
]


def plot(dataset_path: str, output_path: Path, *, show: bool = False) -> None:
    eval_dir = EVAL_ROOT / dataset_path
    csv_paths = [str(path) for path in sorted(eval_dir.glob("*.csv"))]
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"no csv found: {eval_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_csv(csv_paths=csv_paths, plot_name=dataset_path, show=show, output_path=str(output_path))
    print(f"saved: {output_path}")


def plot_agent(index: int, agent_id: str, dataset_cls) -> None:
    dataset = dataset_cls()
    dataset_path = f"{dataset.id}-{agent_id}-v0"
    output_path = PLOT_ROOT / agent_id / f"{index}. {dataset.id}.png"
    plot(dataset_path, output_path, show=SHOW)


if __name__ == "__main__":
    for i, dataset_cls in enumerate(DATASET_LIST, start=1):
        for agent_id in AGENT_LIST:
            print(f"dataset={dataset_cls().id}, agent={agent_id}")
            plot_agent(i, agent_id, dataset_cls)
