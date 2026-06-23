from ice_offline.config.paths import VIEW_ROOT
from ice_offline.run.eval import cal_dataset
from ice_offline.run.eval import cal_main
from ice_offline.run.table import table_mean
from ice_offline.run.table import table_pr95
from ice_offline.run.table import table_true
from test import AGENTS
from test import DATASETS
from test import DATASET_SOURCES

TABLE_BOUNDS = {
    "hopper_d4rl_medium": {"lower": "hopper_random", "upper": "hopper_d4rl_medium"},
    "hopper_replay_medium": {"lower": "hopper_random", "upper": "hopper_replay_medium"},
    "hopper_d4rl_expert": {"lower": "hopper_random", "upper": "hopper_d4rl_expert"},
    "hopper_replay_expert": {"lower": "hopper_random", "upper": "hopper_replay_expert"},
}
TASKS = [
    (dataset_id, env_kwargs, agent_id)
    for dataset_id, _, env_kwargs in DATASETS
    for _, agent_id, _ in AGENTS
]
TABLE_OUTPUT_DIR = VIEW_ROOT / "table" / "experience_init_random"


def unique_items(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def test_task_id(dataset_id: str, agent_id: str, env_kwargs: dict) -> str:
    return f"{dataset_id}-{agent_id}-v0"


def row_id(dataset_id: str, env_kwargs: dict) -> str:
    return dataset_id


def save_test_views(tasks: list[tuple[str, dict, str]]) -> dict[str, tuple[object, object]]:
    outputs: dict[str, tuple[object, object]] = {}
    for dataset_id, env_kwargs, agent_id in tasks:
        task_id = test_task_id(dataset_id, agent_id, env_kwargs)
        returns_output_path, steps_output_path = cal_main(task_id)
        outputs[task_id] = (returns_output_path, steps_output_path)
        print(f"saved: {returns_output_path}")
        print(f"saved: {steps_output_path}")
    return outputs


def save_dataset_views() -> dict[str, tuple[object, object]]:
    outputs: dict[str, tuple[object, object]] = {}
    for dataset_id, source_dataset_id, _ in DATASETS:
        outputs[dataset_id] = cal_dataset(source_dataset_id)
        bounds = TABLE_BOUNDS[source_dataset_id]
        for value in [bounds["lower"], bounds["upper"]]:
            if isinstance(value, str):
                outputs[value] = cal_dataset(value)
    return outputs


def table_inputs(
    tasks: list[tuple[str, dict, str]],
    test_outputs: dict[str, tuple[object, object]],
) -> tuple[list[str], list[str], list[list[object]], list[object], list[object]]:
    dataset_ids = [row_id(dataset_id, env_kwargs) for dataset_id, _, env_kwargs in DATASETS]
    agent_ids = unique_items([agent_id for dataset_id, _, agent_id in tasks if dataset_id in DATASET_SOURCES])
    data_paths: list[list[object]] = []
    lower_values: list[object] = []
    upper_values: list[object] = []

    for dataset_id, source_dataset_id, env_kwargs in DATASETS:
        row: list[object] = [None for _ in agent_ids]
        for task_dataset_id, task_env_kwargs, agent_id in tasks:
            if task_dataset_id != dataset_id or task_env_kwargs != env_kwargs:
                continue
            row[agent_ids.index(agent_id)] = test_outputs[
                test_task_id(dataset_id, agent_id, env_kwargs)
            ][0]
        data_paths.append(row)
        lower_values.append(TABLE_BOUNDS[source_dataset_id]["lower"])
        upper_values.append(TABLE_BOUNDS[source_dataset_id]["upper"])
    return dataset_ids, agent_ids, data_paths, lower_values, upper_values


def plot_table(tasks: list[tuple[str, dict, str]]) -> None:
    test_outputs = save_test_views(tasks)
    dataset_outputs = save_dataset_views()
    dataset_ids, agent_ids, data_paths, lower_values, upper_values = table_inputs(tasks, test_outputs)
    table_true(dataset_ids, agent_ids, data_paths, lower_values, upper_values, dataset_outputs, TABLE_OUTPUT_DIR / "true_returns.csv")
    table_mean(dataset_ids, agent_ids, data_paths, lower_values, upper_values, dataset_outputs, TABLE_OUTPUT_DIR / "mean_returns.csv")
    table_pr95(dataset_ids, agent_ids, data_paths, lower_values, upper_values, dataset_outputs, TABLE_OUTPUT_DIR / "pr95_returns.csv")


if __name__ == "__main__":
    plot_table(TASKS)
