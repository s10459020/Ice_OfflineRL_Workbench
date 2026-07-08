from pathlib import Path

from ice_offline.config.paths import VIEW_ROOT
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import eval_data_path
from ice_offline.config.paths import plot_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.dataset.eval import EvalDataset
from ice_offline.run.boxplot import boxplot_data
from ice_offline.run.boxplot import write_boxplots
from ice_offline.run.eval import EvalRows
from ice_offline.run.eval import eval_returns
from ice_offline.run.eval import eval_steps
from ice_offline.run.eval import write_eval_rows
from ice_offline.run.plot import plot_overlay
from ice_offline.run.table import write_tables

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
    "hopper_d4rl_medium",
    "hopper_d4rl_hybrid",
    "hopper_d4rl_expert",
    "hopper_replay_medium",
    "hopper_replay_expert",
    "walker2d_d4rl_medium",
    "walker2d_d4rl_hybrid",
    "walker2d_d4rl_expert",
    "walker2d_replay_medium",
    "walker2d_replay_expert",
    "halfcheetah_d4rl_medium",
    "halfcheetah_d4rl_hybrid",
    "halfcheetah_d4rl_expert",
    "halfcheetah_replay_medium",
    "halfcheetah_replay_expert",
]

AGENTS = [
    ([None, 0, 50_000], "bc"),
    ([None, 0, 100_000], "td3bc_n"),
    ([None, 0, 200_000], "iql"),
    ([None, 0, 500_000], "cql"),
    # ([None, 0, 200_000], "aspl_gp_punish_005"),
    # ([None, 0, 200_000], "aspl_gp_punish_010"),
    ([None, 0, 200_000], "aspl_gp_punish_050"),
    # ([None, 0, 500_000], "aspl_gp"),
    ([100_000, 0, 500_000], "scas_gp"),
    ([100_000, 0, 500_000], "scaspl_gp"),
]

VALUE_CACHE: dict[str, list[float]] = {}


def _cache(id: str, rows: EvalRows) -> list[float]:
    VALUE_CACHE[id] = [
        value
        for _, values in rows
        for value in values
    ]
    return VALUE_CACHE[id]


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


def _plot_agent_eval(task_id: str, returns_rows: EvalRows) -> Path:
    dataset_id, agent_id, _ = task_id.rsplit("-", 2)
    series_list = [
        (str(step), list(range(1, len(values) + 1)), values)
        for step, values in returns_rows
    ]
    output_path = plot_path("dataset", dataset_id, agent_id)
    path = plot_overlay(task_id, series_list, output_path)
    print(f"saved: {path}")
    return path


def ensure_agent_eval(dataset_id: str, agent_id: str) -> list[float] | None:
    task_id = _task_id(dataset_id, agent_id)
    eval_path = eval_data_path("test", task_id)
    if not eval_path.exists():
        print(f"skip missing: {eval_path}")
        return None

    eval_dataset = EvalDataset(path=eval_path, device="cpu")
    batches = eval_dataset.batch_episodes
    returns_rows = eval_returns(batches)
    steps_rows = eval_steps(batches)
    returns_output_path, steps_output_path = write_eval_rows("test", task_id, returns_rows, steps_rows)
    print(f"saved: {returns_output_path}")
    print(f"saved: {steps_output_path}")
    _plot_agent_eval(task_id, returns_rows)
    return _cache(task_id, returns_rows)


def ensure_dataset_eval(dataset_id: str) -> list[float]:
    return _value(dataset_id)


def ensure_table_datasets(table_specs_list: list[tuple[str, str, str]]) -> None:
    for dataset_id in _ordered_table_dataset_ids(table_specs_list):
        ensure_dataset_eval(dataset_id)


def save_tables(dataset_id_list: list[str], agent_id_list: list[str]) -> tuple[Path, Path, Path]:
    table_specs_list = [spec for spec in TABLES if spec[0] in dataset_id_list]
    dataset_ids, lower_ids, upper_ids = map(list, zip(*table_specs_list))
    data_values = [
        [VALUE_CACHE.get(_task_id(dataset_id, agent_id)) for agent_id in agent_id_list]
        for dataset_id in dataset_ids
    ]
    lower_values = [_value(lower_id) for lower_id in lower_ids]
    upper_values = [_value(upper_id) for upper_id in upper_ids]
    return write_tables(
        "experience_dataset",
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
    output_path = VIEW_ROOT / "boxplot" / "experience_dataset" / "table.png"
    path = boxplot_data("table", members, output_path)
    if path is not None:
        print(f"saved: {path}")
    return path


def save_boxplots(dataset_id_list: list[str], agent_id_list: list[str]) -> list[Path]:
    table_specs_list = [spec for spec in TABLES if spec[0] in dataset_id_list]
    dataset_ids, lower_ids, upper_ids = map(list, zip(*table_specs_list))
    data_values = [
        [VALUE_CACHE.get(_task_id(dataset_id, agent_id)) for agent_id in agent_id_list]
        for dataset_id in dataset_ids
    ]
    lower_values = [_value(lower_id) for lower_id in lower_ids]
    upper_values = [_value(upper_id) for upper_id in upper_ids]
    return write_boxplots(
        "experience_dataset",
        dataset_ids,
        agent_id_list,
        data_values,
        lower_values,
        upper_values,
    )


if __name__ == "__main__":
    for dataset_id, _, _ in TABLES:
        for agent_id in AGENTS:
            ensure_agent_eval(dataset_id, agent_id)

    ensure_table_datasets(TABLES)
    save_tables(DATASETS, AGENTS)
    save_table_boxplot(TABLES)
    save_boxplots(DATASETS, AGENTS)
