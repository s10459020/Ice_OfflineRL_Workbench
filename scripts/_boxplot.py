from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from _return import returns
from _skip import skip_missing_data


PLOT_ROOT = Path("plot")

AGENT_LIST = [
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

DATASET_LIST = [
    ("random", "hopper_random"),
    ("replay", "hopper_replay"),
    ("expert_d4rl", "hopper_expert_d4rl"),
    ("medium_d4rl", "hopper_medium_d4rl"),
    ("medium_replay", "hopper_medium_replay"),
    ("medium_expert", "hopper_medium_expert"),
    ("simple", "hopper_simple"),
    ("medium", "hopper_medium"),
    ("expert", "hopper_expert"),
]

BOTTOM_LIST = [
    "tmps/datasets/d4rl/hopper_random-v2.hdf5",
    "tmps/datasets/d4rl/hopper_random-v2.hdf5",
    "tmps/datasets/d4rl/hopper_random-v2.hdf5",
    "tmps/datasets/d4rl/hopper_random-v2.hdf5",
    "tmps/datasets/d4rl/hopper_random-v2.hdf5",
    "tmps/datasets/d4rl/hopper_random-v2.hdf5",
    "test/hopper_simple-random-v0/data/main_data.hdf5",
    "test/hopper_medium-random-v0/data/main_data.hdf5",
    "test/hopper_expert-random-v0/data/main_data.hdf5",
]

TOP_LIST = [
    "",
    "tmps/datasets/d4rl/hopper_full_replay-v2.hdf5",
    "tmps/datasets/d4rl/hopper_expert-v2.hdf5",
    "tmps/datasets/d4rl/hopper_medium-v2.hdf5",
    "tmps/datasets/d4rl/hopper_medium_replay-v2.hdf5",
    "tmps/datasets/d4rl/hopper_medium_expert-v2.hdf5",
    "mujoco/hopper/simple-v0/data/main_data.hdf5",
    "mujoco/hopper/medium-v0/data/main_data.hdf5",
    "mujoco/hopper/expert-v0/data/main_data.hdf5",
]


def agent_path(dataset_id: str, agent_id: str) -> str:
    return f"test/{dataset_id}-{agent_id}-v0/data/main_data.hdf5"


def add_member(labels: list[str], values: list[list[float]], label: str, dataset_path: str) -> None:
    if not dataset_path:
        return
    if skip_missing_data(dataset_path):
        return
    labels.append(label)
    values.append(returns(dataset_path))


def save_boxplot(index: int, group_name: str, dataset_id: str, bottom_path: str, top_path: str) -> Path:
    labels = []
    values = []

    add_member(labels, values, "random", bottom_path)
    for agent_id in AGENT_LIST:
        add_member(labels, values, agent_id, agent_path(dataset_id, agent_id))
    add_member(labels, values, "dataset", top_path)

    if len(values) == 0:
        print(f"skip empty: {group_name}")
        return PLOT_ROOT / f"{index}. {group_name}.png"

    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_ROOT / f"{index}. {group_name}.png"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(values, tick_labels=labels, showfliers=True)
    ax.set_title(group_name)
    ax.set_ylabel("Return")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"saved: {out_path}")
    return out_path


if __name__ == "__main__":
    for i, ((group_name, dataset_id), bottom_path, top_path) in enumerate(
        zip(DATASET_LIST, BOTTOM_LIST, TOP_LIST),
        start=1,
    ):
        print(f"group={group_name}")
        save_boxplot(i, group_name, dataset_id, bottom_path, top_path)
