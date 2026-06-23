from ice_offline.config.paths import _task_id
from ice_offline.config.paths import VIEW_ROOT
from ice_offline.run.eval import cal_dataset
from ice_offline.run.eval import cal_main
from ice_offline.run.table import table_mean
from ice_offline.run.table import table_pr95
from ice_offline.run.table import table_true

TABLES = {
    "hopper_d4rl_medium": {"lower": "hopper_random", "upper": "hopper_d4rl_medium"},
    "hopper_d4rl_hybrid": {"lower": "hopper_random", "upper": "hopper_d4rl_hybrid"},
    "hopper_d4rl_expert": {"lower": "hopper_random", "upper": "hopper_d4rl_expert"},
    "hopper_replay_medium": {"lower": "hopper_random", "upper": "hopper_replay_medium"},
    "hopper_replay_expert": {"lower": "hopper_random", "upper": "hopper_replay_expert"},
}

TASKS = [
    ("hopper_d4rl_medium", "bc_deterministic"),
    ("hopper_d4rl_hybrid", "bc_deterministic"),
    ("hopper_d4rl_expert", "bc_deterministic"),
    ("hopper_replay_medium", "bc_deterministic"),
    ("hopper_replay_expert", "bc_deterministic"),
    ("hopper_d4rl_medium", "bc_stochastic"),
    ("hopper_d4rl_hybrid", "bc_stochastic"),
    ("hopper_d4rl_expert", "bc_stochastic"),
    ("hopper_replay_medium", "bc_stochastic"),
    ("hopper_replay_expert", "bc_stochastic"),
    ("hopper_d4rl_medium", "td3bc"),
    ("hopper_d4rl_hybrid", "td3bc"),
    ("hopper_d4rl_expert", "td3bc"),
    ("hopper_replay_medium", "td3bc"),
    ("hopper_replay_expert", "td3bc"),
    ("hopper_d4rl_medium", "iql"),
    ("hopper_d4rl_hybrid", "iql"),
    ("hopper_d4rl_expert", "iql"),
    ("hopper_replay_medium", "iql"),
    ("hopper_replay_expert", "iql"),
    ("hopper_replay_medium", "cql"),
    ("hopper_replay_expert", "cql"),
    ("hopper_replay_medium", "aspl"),
    ("hopper_replay_expert", "aspl"),
    ("hopper_d4rl_medium", "sdc"),
    ("hopper_d4rl_hybrid", "sdc"),
    ("hopper_d4rl_expert", "sdc"),
    ("hopper_replay_medium", "sdc"),
    ("hopper_replay_expert", "sdc"),
    ("hopper_d4rl_medium", "sdc_cql"),
    ("hopper_d4rl_hybrid", "sdc_cql"),
    ("hopper_d4rl_expert", "sdc_cql"),
    ("hopper_replay_medium", "sdc_cql"),
    ("hopper_replay_expert", "sdc_cql"),
    ("hopper_d4rl_medium", "scas"),
    ("hopper_d4rl_hybrid", "scas"),
    ("hopper_d4rl_expert", "scas"),
    ("hopper_replay_medium", "scas"),
    ("hopper_replay_expert", "scas"),
    ("hopper_d4rl_medium", "scas_aspl"),
    ("hopper_d4rl_hybrid", "scas_aspl"),
    ("hopper_d4rl_expert", "scas_aspl"),
    ("hopper_replay_medium", "scas_aspl"),
    ("hopper_replay_expert", "scas_aspl"),
]

TABLE_OUTPUT_DIR = VIEW_ROOT / "table"


def unique_items(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def save_test_views(tasks: list[tuple[str, str]]) -> dict[str, tuple[object, object]]:
    outputs: dict[str, tuple[object, object]] = {}
    for dataset_id, agent_id in tasks:
        task_id = _task_id(dataset_id, agent_id)
        returns_output_path, steps_output_path = cal_main(task_id)
        outputs[task_id] = (returns_output_path, steps_output_path)
        print(f"saved: {returns_output_path}")
        print(f"saved: {steps_output_path}")
    return outputs


def save_dataset_views() -> dict[str, tuple[object, object]]:
    outputs: dict[str, tuple[object, object]] = {}
    for dataset_id, bounds in TABLES.items():
        outputs[dataset_id] = cal_dataset(dataset_id)
        for value in [bounds["lower"], bounds["upper"]]:
            if isinstance(value, str):
                outputs[value] = cal_dataset(value)
    return outputs


def table_inputs(tasks: list[tuple[str, str]], test_outputs: dict[str, tuple[object, object]]) -> tuple[list[str], list[str], list[list[object]], list[object], list[object]]:
    dataset_ids = unique_items([dataset_id for dataset_id, _ in tasks if dataset_id in TABLES])
    agent_ids = unique_items([agent_id for dataset_id, agent_id in tasks if dataset_id in TABLES])
    data_paths: list[list[object]] = []
    for dataset_id in dataset_ids:
        row: list[object] = [None for _ in agent_ids]
        for task_dataset_id, agent_id in tasks:
            if task_dataset_id != dataset_id:
                continue
            row[agent_ids.index(agent_id)] = test_outputs[_task_id(dataset_id, agent_id)][0]
        data_paths.append(row)
    lower_values = [TABLES[dataset_id]["lower"] for dataset_id in dataset_ids]
    upper_values = [TABLES[dataset_id]["upper"] for dataset_id in dataset_ids]
    return dataset_ids, agent_ids, data_paths, lower_values, upper_values


def plot_table(tasks: list[tuple[str, str]]) -> None:
    test_outputs = save_test_views(tasks)
    dataset_outputs = save_dataset_views()
    dataset_ids, agent_ids, data_paths, lower_values, upper_values = table_inputs(tasks, test_outputs)
    table_true(dataset_ids, agent_ids, data_paths, lower_values, upper_values, dataset_outputs, TABLE_OUTPUT_DIR / "true_returns.csv")
    table_mean(dataset_ids, agent_ids, data_paths, lower_values, upper_values, dataset_outputs, TABLE_OUTPUT_DIR / "mean_returns.csv")
    table_pr95(dataset_ids, agent_ids, data_paths, lower_values, upper_values, dataset_outputs, TABLE_OUTPUT_DIR / "pr95_returns.csv")


if __name__ == "__main__":
    plot_table(TASKS)
