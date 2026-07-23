import csv
from pathlib import Path

from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import metric_path
from ice_offline.config.paths import table_path
from ice_offline.run.boxplot import write_boxplots
from ice_offline.run.table import table_mean
from ice_offline.run.table import table_var
from view import DATASETS
from view import EXPERIMENT_TRAIN
from view import TABLES
from view import _agent_value
from view import _dataset_value

GROUP = "stability/td3bc"

AGENTS = [
    "td3bc_n",
    "td3bc_plus",
    "td3bc",
    "td3bc_gp_plus",
    "td3bc_gp",
    "td3bc_gpn",
]

TARGET_COUNT = 10


def _agent_target(dataset_id: str, agent_id: str) -> float | None:
    id = experiment_task_id(EXPERIMENT_TRAIN, agent_id, dataset_id)
    path = metric_path(id)
    if not path.exists():
        print(f"skip missing metrics: {path}")
        return None

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        values = [
            float(row["target_q"])
            for row in reader
            if row.get("target_q", "") != ""
        ]

    if not values:
        return None
    targets = values[-TARGET_COUNT:]
    return sum(targets) / len(targets)


def save_mean_targets_group(group: str, dataset_id_list: list[str], agent_id_list: list[str]) -> Path:
    path = table_path(group, "mean_target.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["task", *agent_id_list])
        for dataset_id in dataset_id_list:
            values = [_agent_target(dataset_id, agent_id) for agent_id in agent_id_list]
            writer.writerow([
                dataset_id,
                *["" if value is None else str(float(value)) for value in values],
            ])

    print(f"saved: {path}")
    return path


def save_tables_group(group: str, dataset_id_list: list[str], agent_id_list: list[str]) -> tuple[Path, ...]:
    table_specs_list = [spec for spec in TABLES if spec[0] in dataset_id_list]
    dataset_ids, lower_ids, upper_ids = map(list, zip(*table_specs_list))

    data_values = [
        [_agent_value(dataset_id, agent_id) for agent_id in agent_id_list]
        for dataset_id in dataset_ids
    ]
    lower_values = [_dataset_value(lower_id) for lower_id in lower_ids]
    upper_values = [_dataset_value(upper_id) for upper_id in upper_ids]

    return (
        table_mean(
            dataset_ids,
            agent_id_list,
            data_values,
            lower_values,
            upper_values,
            table_path(group, "mean_returns.csv"),
        ),
        table_var(
            dataset_ids,
            agent_id_list,
            data_values,
            lower_values,
            upper_values,
            table_path(group, "var_returns.csv"),
        ),
        save_mean_targets_group(group, dataset_ids, agent_id_list),
    )


def save_boxplots_group(group: str, dataset_id_list: list[str], agent_id_list: list[str]) -> None:
    table_specs_list = [spec for spec in TABLES if spec[0] in dataset_id_list]
    dataset_ids, lower_ids, upper_ids = map(list, zip(*table_specs_list))

    data_values = [
        [_agent_value(dataset_id, agent_id) for agent_id in agent_id_list]
        for dataset_id in dataset_ids
    ]
    lower_values = [_dataset_value(lower_id) for lower_id in lower_ids]
    upper_values = [_dataset_value(upper_id) for upper_id in upper_ids]

    write_boxplots(
        group,
        dataset_ids,
        agent_id_list,
        data_values,
        lower_values,
        upper_values,
    )


if __name__ == "__main__":
    save_tables_group(GROUP, DATASETS, AGENTS)
    save_boxplots_group(GROUP, DATASETS, AGENTS)
