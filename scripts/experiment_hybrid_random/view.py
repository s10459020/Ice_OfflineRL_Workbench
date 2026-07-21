from pathlib import Path

from ice_offline.config.paths import boxplot_path
from ice_offline.config.paths import eval_path
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import returns_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.analyze import read_csv
from ice_offline.run.boxplot import boxplot_data
from ice_offline.run.boxplot import write_boxplots
from ice_offline.run.table import write_tables
from plot import analyze

EXPERIMENT = "experience_hybrid_random"

TABLES = [
    ("walker2d_random_expert_9", "walker2d_random", "walker2d_d4rl_expert"),
    ("walker2d_random_expert_7", "walker2d_random", "walker2d_d4rl_expert"),
    ("walker2d_random_expert_5", "walker2d_random", "walker2d_d4rl_expert"),
    ("walker2d_random_expert_3", "walker2d_random", "walker2d_d4rl_expert"),
    ("walker2d_random_expert_1", "walker2d_random", "walker2d_d4rl_expert"),
]

DATASETS = [
    "walker2d_random_expert_1",
    "walker2d_random_expert_3",
    "walker2d_random_expert_5",
    "walker2d_random_expert_7",
    "walker2d_random_expert_9",
]

AGENTS = [
    ("bc", None, 50_000),
    ("td3bc_n", None, 100_000),
    ("iql", None, 200_000),
    ("cql", None, 500_000),
    ("aspl_c", None, 500_000),
    ("scas_gp", 500_000, 500_000),
    ("scaspl_n", 500_000, 500_000),
    ("scc_n", 500_000, 500_000),
]

VALUE_CACHE: dict[str, list[float] | None] = {}


def _agent_value(dataset_id: str, agent_id: str) -> list[float] | None:
    key = f"returns:{dataset_id}:{agent_id}"
    if key in VALUE_CACHE:
        return VALUE_CACHE[key]

    id = experiment_task_id(EXPERIMENT, agent_id, dataset_id)
    path = eval_path(id)
    if not path.exists():
        print(f"skip missing eval: {path}")
        VALUE_CACHE[key] = None
        return VALUE_CACHE[key]

    analyze(id, path)
    _, rows = read_csv(returns_path(id))
    VALUE_CACHE[key] = [
        value
        for _, values in rows
        for value in values
    ]
    return VALUE_CACHE[key]


def _dataset_value(dataset_id: str) -> list[float]:
    key = f"returns:{dataset_id}"
    if key in VALUE_CACHE:
        return VALUE_CACHE[key]

    dataset = make_dataset(dataset_id, device="cpu")
    values = [
        float(episode.rewards.sum())
        for episode in dataset.episodes
    ]
    VALUE_CACHE[key] = values
    return values


def _ordered_table_dataset_ids(table_specs_list: list[tuple[str, str, str]]) -> list[str]:
    lowers = [lower_id for _, lower_id, _ in table_specs_list]
    datasets = [dataset_id for dataset_id, _, _ in table_specs_list]
    uppers = [upper_id for _, _, upper_id in table_specs_list]
    dataset_ids: list[str] = []
    for dataset_id in [*lowers, *datasets, *uppers]:
        if dataset_id not in dataset_ids:
            dataset_ids.append(dataset_id)
    return dataset_ids


def save_tables(dataset_id_list: list[str], agent_id_list: list[str]) -> tuple[Path, ...]:
    table_specs_list = [spec for spec in TABLES if spec[0] in dataset_id_list]
    dataset_ids, lower_ids, upper_ids = map(list, zip(*table_specs_list))
    data_values = [
        [_agent_value(dataset_id, agent_id) for agent_id in agent_id_list]
        for dataset_id in dataset_ids
    ]
    lower_values = [_dataset_value(lower_id) for lower_id in lower_ids]
    upper_values = [_dataset_value(upper_id) for upper_id in upper_ids]
    return write_tables(
        EXPERIMENT,
        dataset_ids,
        agent_id_list,
        data_values,
        lower_values,
        upper_values,
    )


def save_boxplots(dataset_id_list: list[str], agent_id_list: list[str]) -> None:
    table_specs_list = [spec for spec in TABLES if spec[0] in dataset_id_list]
    dataset_ids, lower_ids, upper_ids = map(list, zip(*table_specs_list))
    data_values = [
        [_agent_value(dataset_id, agent_id) for agent_id in agent_id_list]
        for dataset_id in dataset_ids
    ]
    lower_values = [_dataset_value(lower_id) for lower_id in lower_ids]
    upper_values = [_dataset_value(upper_id) for upper_id in upper_ids]
    write_boxplots(
        EXPERIMENT,
        dataset_ids,
        agent_id_list,
        data_values,
        lower_values,
        upper_values,
    )


def save_table_boxplots() -> None:
    dataset_ids = _ordered_table_dataset_ids(TABLES)
    boxplot_data(
        "table",
        dataset_ids,
        [_dataset_value(dataset_id) for dataset_id in dataset_ids],
        boxplot_path(EXPERIMENT, "table.png"),
    )


if __name__ == "__main__":
    agent_ids = [agent_id for agent_id, _, _ in AGENTS]
    save_table_boxplots()
    save_tables(DATASETS, agent_ids)
    save_boxplots(DATASETS, agent_ids)
