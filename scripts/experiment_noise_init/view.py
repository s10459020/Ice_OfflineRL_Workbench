from pathlib import Path

from ice_offline.config.paths import eval_path
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import returns_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.analyze import read_csv
from ice_offline.run.boxplot import write_boxplots
from ice_offline.run.table import write_tables
from plot import analyze

EXPERIMENT = "noise_init"

TABLES = [
    ("noise_init_5e-4@walker2d_d4rl_medium", "walker2d_random", "walker2d_d4rl_medium"),
    ("noise_init_5e-3@walker2d_d4rl_medium", "walker2d_random", "walker2d_d4rl_medium"),
    ("noise_init_5e-2@walker2d_d4rl_medium", "walker2d_random", "walker2d_d4rl_medium"),
    ("noise_init_5e-1@walker2d_d4rl_medium", "walker2d_random", "walker2d_d4rl_medium"),
    ("noise_init_5e-4@walker2d_d4rl_hybrid", "walker2d_random", "walker2d_d4rl_medium"),
    ("noise_init_5e-3@walker2d_d4rl_hybrid", "walker2d_random", "walker2d_d4rl_medium"),
    ("noise_init_5e-2@walker2d_d4rl_hybrid", "walker2d_random", "walker2d_d4rl_medium"),
    ("noise_init_5e-1@walker2d_d4rl_hybrid", "walker2d_random", "walker2d_d4rl_medium"),
    ("noise_init_5e-4@walker2d_replay_medium", "walker2d_random", "walker2d_d4rl_medium"),
    ("noise_init_5e-3@walker2d_replay_medium", "walker2d_random", "walker2d_d4rl_medium"),
    ("noise_init_5e-2@walker2d_replay_medium", "walker2d_random", "walker2d_d4rl_medium"),
    ("noise_init_5e-1@walker2d_replay_medium", "walker2d_random", "walker2d_d4rl_medium"),
]

DATASETS = [dataset_id for dataset_id, _, _ in TABLES]

AGENTS = [
    # ("bc", None, 50_000),
    ("td3bc_n", None, 100_000),
    # ("iql", None, 200_000),
    # ("cql", None, 500_000),
    # ("aspl_gp", None, 500_000),
    # ("scas_gp", 100_000, 500_000),
    ("scas_gp", 100_000, 500_000),
    ("scaspl_n", 100_000, 500_000),
]

VALUE_CACHE: dict[str, list[float] | None] = {}


def _task_value(dataset_id: str, agent_id: str) -> list[float] | None:
    key = f"{dataset_id}:{agent_id}"
    if key in VALUE_CACHE:
        return VALUE_CACHE[key]

    task_id = experiment_task_id(EXPERIMENT, agent_id, dataset_id)
    path = eval_path(task_id)
    if not path.exists():
        print(f"skip missing eval: {path}")
        VALUE_CACHE[key] = None
        return VALUE_CACHE[key]

    analyze(task_id, path)
    _, rows = read_csv(returns_path(task_id))
    VALUE_CACHE[key] = [
        value
        for _, values in rows
        for value in values
    ]
    return VALUE_CACHE[key]


def _value(dataset_id: str) -> list[float]:
    if dataset_id in VALUE_CACHE:
        return VALUE_CACHE[dataset_id]  # type: ignore[return-value]
    dataset = make_dataset(dataset_id, device="cpu")
    values = [
        float(episode.rewards.sum())
        for episode in dataset.episodes
    ]
    VALUE_CACHE[dataset_id] = values
    return values


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
        [_task_value(dataset_id, agent_id) for agent_id in agent_id_list]
        for dataset_id in dataset_ids
    ]
    lower_values = [_value(lower_id) for lower_id in lower_ids]
    upper_values = [_value(upper_id) for upper_id in upper_ids]
    return write_boxplots(
        EXPERIMENT,
        dataset_ids,
        agent_id_list,
        data_values,
        lower_values,
        upper_values,
    )


if __name__ == "__main__":
    for dataset_id, _, _ in TABLES:
        for agent_id, _, _ in AGENTS:
            _task_value(dataset_id, agent_id)

    agent_ids = [agent_id for agent_id, _, _ in AGENTS]
    save_tables(DATASETS, agent_ids)
    save_boxplots(DATASETS, agent_ids)
