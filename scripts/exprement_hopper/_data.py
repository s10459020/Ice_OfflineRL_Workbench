import csv
from pathlib import Path

import matplotlib.pyplot as plt
import minari


OUTPUT_ROOT = Path("tmps/data")


def returns(dataset_id: str) -> list[float]:
    dataset = minari.load_dataset(f"{dataset_id}-v0")
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
        "mujoco/hopper/simple",
        "test/hopper_simple_random",
        "test/hopper_simple_bc",
        "test/hopper_simple_cql",
        "test/hopper_simple_iql",
        "test/hopper_simple_aspl",
        "test/hopper_simple_scas",
    )
    ids_medium = (
        "mujoco/hopper/medium",
        "test/hopper_medium_random",
        "test/hopper_medium_bc",
        "test/hopper_medium_cql",
        "test/hopper_medium_iql",
        "test/hopper_medium_aspl",
        "test/hopper_medium_scas",
    )
    ids_expert = (
        "mujoco/hopper/expert",
        "test/hopper_expert_random",
        "test/hopper_expert_bc",
        "test/hopper_expert_cql",
        "test/hopper_expert_iql",
        "test/hopper_expert_aspl",
        "test/hopper_expert_scas",
    )

    save_returns_csv("hopper_simple_returns.csv", *ids_simple)
    save_returns_boxplot("hopper_simple_returns_boxplot.png", *ids_simple)

    save_returns_csv("hopper_medium_returns.csv", *ids_medium)
    save_returns_boxplot("hopper_medium_returns_boxplot.png", *ids_medium)

    save_returns_csv("hopper_expert_returns.csv", *ids_expert)
    save_returns_boxplot("hopper_expert_returns_boxplot.png", *ids_expert)


if __name__ == "__main__":
    main()
