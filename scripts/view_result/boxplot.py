import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ice_offline.dataset.halfcheetah_random import HalfCheetahRandomDataset
from ice_offline.dataset.hopper_random import HopperRandomDataset
from ice_offline.dataset.walker2d_random import Walker2dRandomDataset
from ice_offline.tools.paths import dataset_root


RANDOM_DATASET_CLASS_BY_ENV_NAME = {
    "hopper": HopperRandomDataset,
    "halfcheetah": HalfCheetahRandomDataset,
    "walker2d": Walker2dRandomDataset,
}


def bottom_path(dataset_cls) -> Path:
    if Path(dataset_cls().path).relative_to(dataset_root()).parts[0] == "d4rl":
        random_dataset_cls = RANDOM_DATASET_CLASS_BY_ENV_NAME[dataset_cls().id.split("_", 1)[0]]
        return Path("tmps/returns") / f"{random_dataset_cls().id}-v0.json"
    task_id = f"{dataset_cls().id}-random-v0"
    return Path("tmps/returns") / f"{task_id}.json"


def top_path(dataset_cls) -> Path | None:
    if dataset_cls is RANDOM_DATASET_CLASS_BY_ENV_NAME[dataset_cls().id.split("_", 1)[0]]:
        return None
    return Path("tmps/returns") / f"{dataset_cls().id}-v0.json"


def add_member(labels: list[str], values: list[list[float]], label: str, path: Path | None) -> None:
    if path is None:
        return
    if not path.exists():
        print(f"skip missing: {path}")
        return
    with path.open("r", encoding="utf-8") as file:
        member_values = [float(value) for value in json.load(file)]
    if not member_values:
        return
    labels.append(label)
    values.append(member_values)


def save_boxplot(index: int, dataset_cls, agent_list: list[str]) -> Path:
    group_name = dataset_cls().id
    labels = []
    values = []

    add_member(labels, values, "random", bottom_path(dataset_cls))
    for agent_id in agent_list:
        task_id = f"{dataset_cls().id}-{agent_id}-v0"
        add_member(labels, values, agent_id, Path("tmps/returns") / f"{task_id}.json")
    add_member(labels, values, "dataset", top_path(dataset_cls))

    if len(values) == 0:
        print(f"skip empty: {group_name}")
        return Path("tmps/view") / dataset_cls().env_id / "boxplot" / f"{index}. {dataset_cls().id}.png"

    out_path = Path("tmps/view") / dataset_cls().env_id / "boxplot" / f"{index}. {dataset_cls().id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

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


def main(dataset_class_list: list, agent_id_list: list[str]) -> None:
    agent_list = [agent_id for agent_id in agent_id_list if agent_id != "random"]
    for i, dataset_cls in enumerate(dataset_class_list, start=1):
        group_name = dataset_cls().id
        print(f"group={group_name}")
        save_boxplot(i, dataset_cls, agent_list)


if __name__ == "__main__":
    main([], [])
