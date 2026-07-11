from pathlib import Path

from ice_offline.config.paths import boxplot_path
from ice_offline.config.paths import eval_path
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import returns_path
from ice_offline.dataset._lookup import make_dataset
from plot import analyze
from ice_offline.run.analyze import read_csv
from ice_offline.run.boxplot import boxplot_data
from ice_offline.run.boxplot import write_boxplots
from ice_offline.run.table import write_tables

EXPERIMENT = "base"
EXPERIMENT_TRAIN = "base_train"

TABLES = [
    ("hopper_d4rl_medium", "hopper_random", "hopper_d4rl_medium"),
    ("hopper_d4rl_hybrid", "hopper_random", "hopper_d4rl_hybrid"),
    ("hopper_d4rl_expert", "hopper_random", "hopper_d4rl_expert"),
    ("hopper_replay_medium", "hopper_random", "hopper_d4rl_medium"),
    ("hopper_replay_expert", "hopper_random", "hopper_d4rl_expert"),
    ("walker2d_d4rl_medium", "walker2d_random", "walker2d_d4rl_medium"),
    ("walker2d_d4rl_hybrid", "walker2d_random", "walker2d_d4rl_hybrid"),
    ("walker2d_d4rl_expert", "walker2d_random", "walker2d_d4rl_expert"),
    ("walker2d_replay_medium", "walker2d_random", "walker2d_d4rl_medium"),
    ("walker2d_replay_expert", "walker2d_random", "walker2d_d4rl_expert"),
    ("halfcheetah_d4rl_medium", "halfcheetah_random", "halfcheetah_d4rl_medium"),
    ("halfcheetah_d4rl_hybrid", "halfcheetah_random", "halfcheetah_d4rl_hybrid"),
    ("halfcheetah_d4rl_expert", "halfcheetah_random", "halfcheetah_d4rl_expert"),
    ("halfcheetah_replay_medium", "halfcheetah_random", "halfcheetah_d4rl_medium"),
    ("halfcheetah_replay_expert", "halfcheetah_random", "halfcheetah_d4rl_expert"),
]

DATASETS = [
    # "hopper_d4rl_medium",
    # "hopper_d4rl_hybrid",
    # "hopper_d4rl_expert",
    # "hopper_replay_medium",
    # "hopper_replay_expert",
    "walker2d_d4rl_medium",
    "walker2d_d4rl_hybrid",
    "walker2d_d4rl_expert",
    "walker2d_replay_medium",
    "walker2d_replay_expert",
    # "halfcheetah_d4rl_medium",
    # "halfcheetah_d4rl_hybrid",
    # "halfcheetah_d4rl_expert",
    # "halfcheetah_replay_medium",
    # "halfcheetah_replay_expert",
]

AGENTS = [
    ([None, 0, 50_000], "bc"),
    ([None, 0, 100_000], "td3bc_n"),
    ([None, 0, 200_000], "iql"),
    ([None, 0, 500_000], "cql"),
    ([None, 0, 500_000], "aspl_gp"),
    ([100_000, 0, 500_000], "scc"),
    ([100_000, 0, 500_000], "scc_ns"),
    ([100_000, 0, 500_000], "scc_n"),
    ([100_000, 0, 500_000], "scc_gp"),
    ([100_000, 0, 500_000], "scas_n"),
    ([100_000, 0, 500_000], "scas_n_lambda_0"),
    ([100_000, 0, 500_000], "scas_n_lambda_100"),
    ([100_000, 0, 500_000], "scas_gp"),
    ([100_000, 0, 500_000], "scaspl_n"),
    ([100_000, 0, 500_000], "scaspl_n_lambda_0"),
    ([100_000, 0, 500_000], "scaspl_n_lambda_100"),
    ([100_000, 0, 500_000], "scaspl_ns"),
    ([100_000, 0, 500_000], "scaspl_gp"),
]

VALUE_CACHE: dict[str, list[float]] = {}

def _agent_value(dataset_id: str, agent_id: str) -> list[float]:
    key = f"{dataset_id}:{agent_id}"
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
    key = f"{dataset_id}"
    if key in VALUE_CACHE:
        return VALUE_CACHE[key]
    
    dataset = make_dataset(dataset_id, device="cpu")
    values = [
        float(episode.rewards.sum())
        for episode in dataset.episodes
    ]

    VALUE_CACHE[key] = values
    return VALUE_CACHE[key]

def save_tables(dataset_id_list: list[str], agent_id_list: list[str]) -> tuple[Path, Path, Path]:
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
    datasets, lowers, uppers = zip(*table_specs_list)
    
    data_values = [
        [_agent_value(dataset_id, agent_id) for agent_id in agent_id_list]
        for dataset_id in datasets
    ]
    lower_values = [_dataset_value(lower_id) for lower_id in lowers]
    upper_values = [_dataset_value(upper_id) for upper_id in uppers]
    
    write_boxplots(
        EXPERIMENT,
        datasets,
        agent_id_list,
        data_values,
        lower_values,
        upper_values,
    )

def save_table_boxplots() -> None:
    datasets, lowers, uppers = zip(*TABLES)
    dataset_ids: list[str] = []
    for dataset_id in [*lowers, *datasets, *uppers]:
        if dataset_id not in dataset_ids:
            dataset_ids.append(dataset_id)

    boxplot_data(
        "table",
        dataset_ids,
        [_dataset_value(dataset_id) for dataset_id in dataset_ids],
        boxplot_path(EXPERIMENT, "table.png"),
    )


if __name__ == "__main__":
    agent_ids = [agent_id for _, agent_id in AGENTS]
    save_table_boxplots()
    save_tables(DATASETS, agent_ids)
    save_boxplots(DATASETS, agent_ids)
