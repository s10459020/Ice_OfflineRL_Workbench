import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ice_offline.data.minari.loader import MinariLoader
from ice_offline.tools.paths import minari_root


RETURNS_ROOT = Path("result/returns")
PLOT_ROOT = Path("plot")

DATASET_LIST = [
    ("simple", "hopper_simple"),
    ("medium", "hopper_medium"),
    ("expert", "hopper_expert"),
    ("random", "hopper_random"),
    ("replay", "hopper_replay"),
    ("medium_d4rl", "hopper_medium_d4rl"),
    ("medium_replay", "hopper_medium_replay"),
    ("medium_expert", "hopper_medium_expert"),
]

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
    "random",
]


def returns_path(dataset_path: str) -> Path:
    return RETURNS_ROOT / f"{dataset_path.replace('/', '__')}.json"


def compute_returns(dataset_path: str) -> list[float]:
    dataset = MinariLoader(minari_root() / dataset_path)
    values = []
    for episode in dataset.iterate_episodes():
        values.append(float(episode.rewards.sum()))
    return values


def returns(dataset_path: str) -> list[float]:
    path = returns_path(dataset_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            return [float(value) for value in json.load(file)]

    values = compute_returns(dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(values, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"saved: {path}")
    return values


def dataset_path(dataset_id: str, agent_id: str) -> str:
    return f"test/{dataset_id}-{agent_id}-v0/data/main_data.hdf5"


def save_boxplot(index: int, group_name: str, dataset_id: str) -> Path:
    labels = AGENT_LIST
    values = [returns(dataset_path(dataset_id, agent_id)) for agent_id in AGENT_LIST]

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
    for i, (group_name, dataset_id) in enumerate(DATASET_LIST, start=1):
        print(f"group={group_name}")
        save_boxplot(i, group_name, dataset_id)
