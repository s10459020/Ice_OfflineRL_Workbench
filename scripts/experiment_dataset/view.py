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


EXPERIMENT = "in_dataset"

TABLES = [
    ("hopper_d4rl_medium", "hopper_random", "hopper_d4rl_medium"),
    ("hopper_d4rl_expert", "hopper_random", "hopper_d4rl_expert"),
    ("hopper_d4rl_hybrid", "hopper_random", "hopper_d4rl_hybrid"),
    ("hopper_replay_medium", "hopper_random", "hopper_d4rl_medium"),
    ("hopper_replay_expert", "hopper_random", "hopper_d4rl_expert"),
    ("walker2d_d4rl_medium", "walker2d_random", "walker2d_d4rl_medium"),
    ("walker2d_d4rl_expert", "walker2d_random", "walker2d_d4rl_expert"),
    ("walker2d_d4rl_hybrid", "walker2d_random", "walker2d_d4rl_hybrid"),
    ("walker2d_replay_medium", "walker2d_random", "walker2d_d4rl_medium"),
    ("walker2d_replay_expert", "walker2d_random", "walker2d_d4rl_expert"),
    ("halfcheetah_d4rl_medium", "halfcheetah_random", "halfcheetah_d4rl_medium"),
    ("halfcheetah_d4rl_expert", "halfcheetah_random", "halfcheetah_d4rl_expert"),
    ("halfcheetah_d4rl_hybrid", "halfcheetah_random", "halfcheetah_d4rl_hybrid"),
    ("halfcheetah_replay_medium", "halfcheetah_random", "halfcheetah_d4rl_medium"),
    ("halfcheetah_replay_expert", "halfcheetah_random", "halfcheetah_d4rl_expert"),
]

DATASETS = [
    "hopper_d4rl_medium",
    "hopper_d4rl_expert",
    "hopper_d4rl_hybrid",
    "hopper_replay_medium",
    "hopper_replay_expert",
    "walker2d_d4rl_medium",
    "walker2d_d4rl_expert",
    "walker2d_d4rl_hybrid",
    "walker2d_replay_medium",
    "walker2d_replay_expert",
    "halfcheetah_d4rl_medium",
    "halfcheetah_d4rl_expert",
    "halfcheetah_d4rl_hybrid",
    "halfcheetah_replay_medium",
    "halfcheetah_replay_expert",
]

AGENTS = [
    ([None, 0, 50_000], "bc"),
    ([None, 0, 100_000], "td3bc_n"),
    ([None, 0, 200_000], "iql"),
    ([None, 0, 500_000], "cql"),
    ([None, 0, 200_000], "aspl_gp_punish_050"),
    ([100_000, 0, 500_000], "scas_gp"),
    ([100_000, 0, 500_000], "scaspl_gp"),
]

VALUE_CACHE: dict[str, list[float]] = {}


def _task_value(dataset_id: str, agent_id: str) -> list[float]:
    task_id = _task_id(dataset_id, agent_id)
    if task_id not in VALUE_CACHE:
        rows = eval(task_id, eval_data_path(EXPERIMENT, task_id))
        VALUE_CACHE[task_id] = [
            value
            for _, values in rows
            for value in values
        ]
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


def save_tables(dataset_id_list: list[str], agent_id_list: list[str]) -> tuple[Path, ...]:
    table_specs_list = [spec for spec in TABLES if spec[0] in dataset_id_list]
    dataset_ids, lower_ids, upper_ids = map(list, zip(*table_specs_list))
    data_values = [
        [_task_value(dataset_id, agent_id) for agent_id in agent_id_list]
        for dataset_id in dataset_ids
    ]
    lower_values = [_value(lower_id) for lower_id in lower_ids]
    upper_values = [_value(upper_id) for upper_id in upper_ids]
    return write_tables(
        "experience_a",
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
    output_path = VIEW_ROOT / "boxplot" / "experience_a" / "table.png"
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
        "experience_a",
        dataset_ids,
        agent_id_list,
        data_values,
        lower_values,
        upper_values,
    )


if __name__ == "__main__":
    agent_ids = [agent_id for _, agent_id in AGENTS]
    for dataset_id, _, _ in TABLES:
        for agent_id in agent_ids:
            _task_value(dataset_id, agent_id)

    save_tables(DATASETS, agent_ids)
    save_table_boxplot(TABLES)
    save_boxplots(DATASETS, agent_ids)
