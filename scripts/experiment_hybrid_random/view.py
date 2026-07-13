from pathlib import Path

from ice_offline.config.paths import VIEW_ROOT
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import eval_data_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.boxplot import boxplot_data
from ice_offline.run.boxplot import write_boxplots
from ice_offline.run.eval import EvalRows
from ice_offline.run.table import write_tables
from plot import eval

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
    "scaspl",
]

VALUE_CACHE: dict[str, list[float]] = {}


def _cache(id: str, rows: EvalRows) -> list[float]:
    VALUE_CACHE[id] = [
        value
        for _, values in rows
        for value in values
    ]
    return VALUE_CACHE[id]


def _task_value(dataset_id: str, agent_id: str) -> list[float]:
    task_id = _task_id(dataset_id, agent_id)
    if task_id not in VALUE_CACHE:
        _cache(task_id, eval(task_id, eval_data_path("test", task_id)))
    return VALUE_CACHE[task_id]


def _ordered_table_dataset_ids(table_specs_list: list[tuple[str, str, str]]) -> list[str]:
    lowers = [lower_id for _, lower_id, _ in table_specs_list]
    datasets = [dataset_id for dataset_id, _, _ in table_specs_list]
    uppers = [upper_id for _, _, upper_id in table_specs_list]
    dataset_ids: list[str] = []
    for dataset_id in [*lowers, *datasets, *uppers]:
        if dataset_id not in dataset_ids:
            dataset_ids.append(dataset_id)
    return dataset_ids


def _value(dataset_id: str) -> list[float]:
    if dataset_id in VALUE_CACHE:
        return VALUE_CACHE[dataset_id]
    dataset = make_dataset(dataset_id, device="cpu")
    values = [
        float(episode.rewards.sum())
        for episode in dataset.episodes
    ]
    VALUE_CACHE[dataset_id] = values
    return values


def ensure_dataset_eval(dataset_id: str) -> list[float]:
    return _value(dataset_id)


def ensure_table_datasets(table_specs_list: list[tuple[str, str, str]]) -> None:
    for dataset_id in _ordered_table_dataset_ids(table_specs_list):
        ensure_dataset_eval(dataset_id)


def save_tables(dataset_id_list: list[str], agent_id_list: list[str]) -> tuple[Path, Path, Path, Path]:
    table_specs_list = [spec for spec in TABLES if spec[0] in dataset_id_list]
    dataset_ids, lower_ids, upper_ids = map(list, zip(*table_specs_list))
    data_values = [
        [_task_value(dataset_id, agent_id) for agent_id in agent_id_list]
        for dataset_id in dataset_ids
    ]
    lower_values = [_value(lower_id) for lower_id in lower_ids]
    upper_values = [_value(upper_id) for upper_id in upper_ids]
    return write_tables(
        "experience_hybrid_random",
        dataset_ids,
        agent_id_list,
        data_values,
        lower_values,
        upper_values,
    )


def save_table_boxplot(table_specs_list: list[tuple[str, str, str]]) -> Path | None:
    members = [
        (dataset_id, _value(dataset_id))
        for dataset_id in _ordered_table_dataset_ids(table_specs_list)
    ]
    output_path = VIEW_ROOT / "boxplot" / "experience_hybrid_random" / "table.png"
    path = boxplot_data("table", members, output_path)
    if path is not None:
        print(f"saved: {path}")
    return path


def save_boxplots(dataset_id_list: list[str], agent_id_list: list[str]) -> list[Path]:
    table_specs_list = [spec for spec in TABLES if spec[0] in dataset_id_list]
    dataset_ids, lower_ids, upper_ids = map(list, zip(*table_specs_list))
    data_values = [
        [_task_value(dataset_id, agent_id) for agent_id in agent_id_list]
        for dataset_id in dataset_ids
    ]
    lower_values = [_value(lower_id) for lower_id in lower_ids]
    upper_values = [_value(upper_id) for upper_id in upper_ids]
    return write_boxplots(
        "experience_hybrid_random",
        dataset_ids,
        agent_id_list,
        data_values,
        lower_values,
        upper_values,
    )


if __name__ == "__main__":
    ensure_table_datasets(TABLES)
    for dataset_id, _, _ in TABLES:
        for agent_id in AGENTS:
            _task_value(dataset_id, agent_id)

    save_tables(DATASETS, AGENTS)
    save_table_boxplot(TABLES)
    save_boxplots(DATASETS, AGENTS)
