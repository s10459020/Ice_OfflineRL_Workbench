import csv
from pathlib import Path

import matplotlib.pyplot as plt
import minari
import numpy as np

from ice_offline.data.d4rl.loader import D4rlLoader


OUTPUT_ROOT = Path("tmps/data")


def returns(dataset_id: str) -> list[float]:
    dataset = minari.load_dataset(dataset_id)
    values = []
    for episode in dataset.iterate_episodes():
        values.append(float(episode.rewards.sum()))
    return values


def save_returns_csv(csv_name: str, *dataset_ids: str) -> Path:
    data = {dataset_id: returns(dataset_id) for dataset_id in dataset_ids}
    max_len = max(len(values) for values in data.values())

    rows = []
    for i in range(max_len):
        row = [i + 1]
        for dataset_id in dataset_ids:
            values = data[dataset_id]
            row.append(values[i] if i < len(values) else "")
        rows.append(row)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / csv_name
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", *dataset_ids])
        writer.writerows(rows)
    print(f"saved: {out_path}")
    return out_path


def save_returns_boxplot(plot_name: str, *dataset_ids: str) -> Path:
    data = {dataset_id: returns(dataset_id) for dataset_id in dataset_ids}
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


def main() -> None:
    ids_simple = (
        "mujoco/hopper/simple-v0",
        "test/hopper_simple_random-v0",
        "test/hopper_simple_bc-v0",
        "test/hopper_simple_cql-v0",
        "test/hopper_simple_iql-v0",
        "test/hopper_simple_aspl-v0",
        "test/hopper_simple_scas-v0",
    )
    ids_medium = (
        "mujoco/hopper/medium-v0",
        "test/hopper_medium_random-v0",
        "test/hopper_medium_bc-v0",
        "test/hopper_medium_cql-v0",
        "test/hopper_medium_iql-v0",
        "test/hopper_medium_aspl-v0",
        "test/hopper_medium_scas-v0",
    )
    ids_expert = (
        "mujoco/hopper/expert-v0",
        "test/hopper_expert_random-v0",
        "test/hopper_expert_bc-v0",
        "test/hopper_expert_cql-v0",
        "test/hopper_expert_iql-v0",
        "test/hopper_expert_aspl-v0",
        "test/hopper_expert_scas-v0",
    )
    ids_medium_replay = (
        "test/hopper_medium_replay_d4rl_random-v0",
        "test/hopper_medium_replay_d4rl_bc-v0",
        "test/hopper_medium_replay_d4rl_cql-v0",
        "test/hopper_medium_replay_d4rl_iql-v0",
        "test/hopper_medium_replay_d4rl_aspl-v0",
        "test/hopper_medium_replay_d4rl_scas-v0",
    )
    ids_medium_expert = (
        "test/hopper_medium_expert_d4rl_random-v0",
        "test/hopper_medium_expert_d4rl_bc-v0",
        "test/hopper_medium_expert_d4rl_cql-v0",
        "test/hopper_medium_expert_d4rl_iql-v0",
        "test/hopper_medium_expert_d4rl_aspl-v0",
        "test/hopper_medium_expert_d4rl_scas-v0",
    )

    #save_returns_csv("hopper_simple_returns.csv", *ids_simple)
    #save_returns_boxplot("hopper_simple_returns_boxplot.png", *ids_simple)

    #save_returns_csv("hopper_medium_returns.csv", *ids_medium)
    #save_returns_boxplot("hopper_medium_returns_boxplot.png", *ids_medium)

    #save_returns_csv("hopper_expert_returns.csv", *ids_expert)
    #save_returns_boxplot("hopper_expert_returns_boxplot.png", *ids_expert)

    save_returns_csv("hopper_medium_replay_returns.csv", *ids_medium_replay)
    save_returns_boxplot("hopper_medium_replay_returns_boxplot.png", *ids_medium_replay)

    save_returns_csv("hopper_medium_expert_returns.csv", *ids_medium_expert)
    save_returns_boxplot("hopper_medium_expert_returns_boxplot.png", *ids_medium_expert)


if __name__ == "__main__":
    main()
