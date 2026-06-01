import csv
from pathlib import Path

import matplotlib.pyplot as plt

from ice_offline.data.minari.loader import MinariLoader
from ice_offline.tools.paths import minari_root


OUTPUT_ROOT = Path("tmps/data")


def returns(dataset_path: str) -> list[float]:
    data_path = minari_root() / dataset_path
    dataset = MinariLoader(data_path)
    values = []
    for episode in dataset.iterate_episodes():
        values.append(float(episode.rewards.sum()))
    return values


def save_returns_csv(csv_name: str, *dataset_paths: str) -> Path:
    data = {dataset_path: returns(dataset_path) for dataset_path in dataset_paths}
    max_len = max(len(values) for values in data.values())

    rows = []
    for i in range(max_len):
        row = [i + 1]
        for dataset_path in dataset_paths:
            values = data[dataset_path]
            row.append(values[i] if i < len(values) else "")
        rows.append(row)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / csv_name
    with out_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", *dataset_paths])
        writer.writerows(rows)
    print(f"saved: {out_path}")
    return out_path


def save_returns_boxplot(plot_name: str, *dataset_paths: str) -> Path:
    data = {dataset_path: returns(dataset_path) for dataset_path in dataset_paths}
    labels = list(data.keys())
    values = [data[label] for label in labels]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / plot_name

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(values, tick_labels=labels, showfliers=True)
    ax.set_title(plot_name.replace(".png", ""))
    ax.set_ylabel("Return")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"saved: {out_path}")
    return out_path


IDS_SIMPLE = (
    "mujoco/hopper/simple-v0/data/main_data.hdf5",
    "test/hopper_simple-random-v0/data/main_data.hdf5",
    "test/hopper_simple-bc_deterministic-v0/data/main_data.hdf5",
    "test/hopper_simple_cql-v0/data/main_data.hdf5",
    "test/hopper_simple-iql-v0/data/main_data.hdf5",
    "test/hopper_simple-aspl-v0/data/main_data.hdf5",
    "test/hopper_simple-scas-v0/data/main_data.hdf5",
)
IDS_MEDIUM = (
    "mujoco/hopper/medium-v0/data/main_data.hdf5",
    "test/hopper_medium-random-v0/data/main_data.hdf5",
    "test/hopper_medium-bc_deterministic-v0/data/main_data.hdf5",
    "test/hopper_medium_cql-v0/data/main_data.hdf5",
    "test/hopper_medium-iql-v0/data/main_data.hdf5",
    "test/hopper_medium-aspl-v0/data/main_data.hdf5",
    "test/hopper_medium-scas-v0/data/main_data.hdf5",
)
IDS_EXPERT = (
    "mujoco/hopper/expert-v0/data/main_data.hdf5",
    "test/hopper_expert-random-v0/data/main_data.hdf5",
    "test/hopper_expert-bc_deterministic-v0/data/main_data.hdf5",
    "test/hopper_expert_cql-v0/data/main_data.hdf5",
    "test/hopper_expert-iql-v0/data/main_data.hdf5",
    "test/hopper_expert-aspl-v0/data/main_data.hdf5",
    "test/hopper_expert-scas-v0/data/main_data.hdf5",
)
IDS_MEDIUM_REPLAY = (
    "test/hopper_medium_replay-random-v0/data/main_data.hdf5",
    "test/hopper_medium_replay-bc_deterministic-v0/data/main_data.hdf5",
    "test/hopper_medium_replay_d4rl_cql-v0/data/main_data.hdf5",
    "test/hopper_medium_replay-iql-v0/data/main_data.hdf5",
    "test/hopper_medium_replay-aspl-v0/data/main_data.hdf5",
    "test/hopper_medium_replay-scas-v0/data/main_data.hdf5",
)
IDS_MEDIUM_EXPERT = (
    "test/hopper_medium_expert-random-v0/data/main_data.hdf5",
    "test/hopper_medium_expert-bc_deterministic-v0/data/main_data.hdf5",
    "test/hopper_medium_expert_d4rl_cql-v0/data/main_data.hdf5",
    "test/hopper_medium_expert-iql-v0/data/main_data.hdf5",
    "test/hopper_medium_expert-aspl-v0/data/main_data.hdf5",
    "test/hopper_medium_expert-scas-v0/data/main_data.hdf5",
)


if __name__ == "__main__":
    # save_returns_csv("hopper_simple-returns.csv", *IDS_SIMPLE)
    # save_returns_boxplot("hopper_simple-returns_boxplot.png", *IDS_SIMPLE)
    # save_returns_csv("hopper_medium-returns.csv", *IDS_MEDIUM)
    # save_returns_boxplot("hopper_medium-returns_boxplot.png", *IDS_MEDIUM)
    # save_returns_csv("hopper_expert-returns.csv", *IDS_EXPERT)
    # save_returns_boxplot("hopper_expert-returns_boxplot.png", *IDS_EXPERT)
    save_returns_csv("hopper_medium_replay_returns.csv", *IDS_MEDIUM_REPLAY)
    save_returns_boxplot("hopper_medium_replay_returns_boxplot.png", *IDS_MEDIUM_REPLAY)
    save_returns_csv("hopper_medium_expert_returns.csv", *IDS_MEDIUM_EXPERT)
    save_returns_boxplot("hopper_medium_expert_returns_boxplot.png", *IDS_MEDIUM_EXPERT)
