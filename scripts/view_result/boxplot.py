from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from view_result.returns import returns
from view_result.skip import skip_missing_data
from view_result.task import AGENT_ID_LIST
from view_result.task import DATASET_CLASS_LIST
from view_result.task import boxplot_output_path
from view_result.task import bottom_dataset_path
from view_result.task import dataset_group_name
from view_result.task import test_dataset_path
from view_result.task import top_dataset_path


AGENT_LIST = [agent_id for agent_id in AGENT_ID_LIST if agent_id != "random"]


def add_member(labels: list[str], values: list[list[float]], label: str, dataset_path: str) -> None:
    if not dataset_path:
        return
    if skip_missing_data(dataset_path):
        return
    labels.append(label)
    values.append(returns(dataset_path))


def save_boxplot(index: int, dataset_cls) -> Path:
    group_name = dataset_group_name(dataset_cls)
    bottom_path = bottom_dataset_path(dataset_cls)
    top_path = top_dataset_path(dataset_cls)
    labels = []
    values = []

    add_member(labels, values, "random", bottom_path)
    for agent_id in AGENT_LIST:
        add_member(labels, values, agent_id, test_dataset_path(dataset_cls, agent_id))
    add_member(labels, values, "dataset", top_path)

    if len(values) == 0:
        print(f"skip empty: {group_name}")
        return boxplot_output_path(index, dataset_cls)

    out_path = boxplot_output_path(index, dataset_cls)
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


def main() -> None:
    for i, dataset_cls in enumerate(DATASET_CLASS_LIST, start=1):
        group_name = dataset_group_name(dataset_cls)
        print(f"group={group_name}")
        save_boxplot(i, dataset_cls)


if __name__ == "__main__":
    main()
