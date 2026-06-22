from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import VIEW_ROOT
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.eval import cal_dataset
from ice_offline.run.eval import cal_main
from ice_offline.run.table import table_mean
from ice_offline.run.table import table_pr95
from ice_offline.run.table import table_true
from ice_offline.run.test import test

TABLES = {
    # "hopper_one_simple": {"lower": "hopper_random", "upper": "hopper_one_simple"},
    # "hopper_random": {"lower": "hopper_random", "upper": 1000.0},
    # "hopper_replay_expert": {"lower": "hopper_random", "upper": "hopper_replay_expert"},
    # "hopper_replay_medium": {"lower": "hopper_random", "upper": "hopper_replay_medium"},
    "hopper_d4rl_medium": {"lower": "hopper_random", "upper": "hopper_d4rl_medium"},
    "hopper_d4rl_hybrid": {"lower": "hopper_random", "upper": "hopper_d4rl_hybrid"},
    "hopper_d4rl_expert": {"lower": "hopper_random", "upper": "hopper_d4rl_expert"},
    # "hopper_simple": {"lower": "hopper_random", "upper": "hopper_simple"},
    # "hopper_medium": {"lower": "hopper_random", "upper": "hopper_medium"},
    # "hopper_expert": {"lower": "hopper_random", "upper": "hopper_expert"},
}

TASKS = [
    # ({"agent_step": 20_000, "episodes": 5}, "hopper_one_simple", {"reset_noise_scale": 0.0}, "bc_deterministic", {}),
    # ({"agent_step": 20_000, "episodes": 5}, "hopper_one_simple", {"reset_noise_scale": 0.0}, "bc_stochastic", {}),
    # ({"agent_step": 20_000, "episodes": 5}, "hopper_one_simple", {"reset_noise_scale": 0.0}, "td3bc", {}),
    # ({"agent_step": 50_000, "episodes": 5}, "hopper_one_simple", {"reset_noise_scale": 0.0}, "cql", {}),
    # ({"agent_step": 50_000, "episodes": 5}, "hopper_one_simple", {"reset_noise_scale": 0.0}, "aspl", {"alpha": 0.5}),
    ({"agent_step": 200_000}, "hopper_d4rl_medium", {}, "bc_deterministic", {}),
    ({"agent_step": 200_000}, "hopper_d4rl_hybrid", {}, "bc_deterministic", {}),
    ({"agent_step": 200_000}, "hopper_d4rl_expert", {}, "bc_deterministic", {}),
    ({"agent_step": 200_000}, "hopper_d4rl_medium", {}, "bc_stochastic", {}),
    ({"agent_step": 200_000}, "hopper_d4rl_hybrid", {}, "bc_stochastic", {}),
    ({"agent_step": 200_000}, "hopper_d4rl_expert", {}, "bc_stochastic", {}),
    ({"agent_step": 200_000}, "hopper_d4rl_medium", {}, "td3bc", {}),
    ({"agent_step": 200_000}, "hopper_d4rl_hybrid", {}, "td3bc", {}),
    ({"agent_step": 200_000}, "hopper_d4rl_expert", {}, "td3bc", {}),
    ({"agent_step": 500_000}, "hopper_d4rl_medium", {}, "cql_soft_q", {"threshold": 1.5}),
    ({"agent_step": 500_000}, "hopper_d4rl_hybrid", {}, "cql_soft_q", {"threshold": 1.5}),
    ({"agent_step": 500_000}, "hopper_d4rl_expert", {}, "cql_soft_q", {"threshold": 1.0}),
    ({"agent_step": 500_000}, "hopper_d4rl_medium", {}, "aspl", {"alpha": 0.5}),
    ({"agent_step": 500_000}, "hopper_d4rl_hybrid", {}, "aspl", {"alpha": 0.5}),
    ({"agent_step": 500_000}, "hopper_d4rl_expert", {}, "aspl", {"alpha": 1}),
    ({"agent_step": 500_000}, "hopper_d4rl_medium", {}, "sdc_cql", {"threshold": 10}),
    ({"agent_step": 500_000}, "hopper_d4rl_hybrid", {}, "sdc_cql", {"threshold": 5}),
    ({"agent_step": 500_000}, "hopper_d4rl_expert", {}, "sdc_cql", {"threshold": 0.5}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_d4rl_medium", {}, "scas_min", {}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_d4rl_hybrid", {}, "scas_min", {}),
    ({"agent_step": 500_000, "model_step": 100_000}, "hopper_d4rl_expert", {}, "scas_min", {}),
]

TASK_KWARGS = {
    # "episodes": 100,
}

DATASETS = [
    # ("hopper_one_simple", {}),
    # ("hopper_simple", {}),
    # ("hopper_medium", {}),
    # ("hopper_expert", {}),
]


AGENTS = [
    # ("bc_stochastic", {}),
    # ("aspl", {"alpha": 0.5}),
    # ("bc_deterministic", {}),
    # ("td3bc", {}),
    # ("cql", {}),
]


TABLE_OUTPUT_DIR = VIEW_ROOT / "table"


def unique_items(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def normalize_tasks() -> list[tuple[dict[str, object], str, dict[str, object], str, dict[str, object]]]:
    if TASKS:
        return TASKS
    return [
        (
            dict(TASK_KWARGS),
            dataset_id,
            env_kwargs,
            agent_id,
            agent_kwargs,
        )
        for dataset_id, env_kwargs in DATASETS
        for agent_id, agent_kwargs in AGENTS
    ]


def save_test_views(tasks: list[tuple[dict[str, object], str, dict[str, object], str, dict[str, object]]]) -> None:
    outputs: dict[str, tuple[object, object]] = {}
    for _, dataset_id, _, agent_id, _ in tasks:
        task_id = _task_id(dataset_id, agent_id)
        returns_output_path, steps_output_path = cal_main(task_id)
        outputs[task_id] = (returns_output_path, steps_output_path)
        print(f"saved: {returns_output_path}")
        print(f"saved: {steps_output_path}")
    return outputs


def save_dataset_views() -> None:
    outputs: dict[str, tuple[object, object]] = {}
    for dataset_id, bounds in TABLES.items():
        outputs[dataset_id] = cal_dataset(dataset_id)
        for value in [bounds["lower"], bounds["upper"]]:
            if isinstance(value, str):
                outputs[value] = cal_dataset(value)
    return outputs


def _table_inputs(
    tasks: list[tuple[dict[str, object], str, dict[str, object], str, dict[str, object]]],
    test_outputs: dict[str, tuple[object, object]],
):
    dataset_ids = list(TABLES.keys())
    agent_ids = unique_items([agent_id for _, _, _, agent_id, _ in tasks])
    data_paths: list[list[object]] = []
    for dataset_id in dataset_ids:
        row: list[object] = [None for _ in agent_ids]
        for _, task_dataset_id, _, agent_id, _ in tasks:
            if task_dataset_id != dataset_id:
                continue
            row[agent_ids.index(agent_id)] = test_outputs[_task_id(dataset_id, agent_id)][0]
        data_paths.append(row)
    lower_values = [TABLES[dataset_id]["lower"] for dataset_id in dataset_ids]
    upper_values = [TABLES[dataset_id]["upper"] for dataset_id in dataset_ids]
    return dataset_ids, agent_ids, data_paths, lower_values, upper_values


def view_test(tasks: list[tuple[dict[str, object], str, dict[str, object], str, dict[str, object]]]) -> None:
    test_outputs = save_test_views(tasks)
    dataset_outputs = save_dataset_views()
    dataset_ids, agent_ids, data_paths, lower_values, upper_values = _table_inputs(tasks, test_outputs)
    table_true(dataset_ids, agent_ids, data_paths, lower_values, upper_values, dataset_outputs, TABLE_OUTPUT_DIR / "true_returns.csv")
    table_mean(dataset_ids, agent_ids, data_paths, lower_values, upper_values, dataset_outputs, TABLE_OUTPUT_DIR / "mean_returns.csv")
    table_pr95(dataset_ids, agent_ids, data_paths, lower_values, upper_values, dataset_outputs, TABLE_OUTPUT_DIR / "pr95_returns.csv")


def main() -> None:
    tasks = normalize_tasks()

    for task_kwargs, dataset_id, env_kwargs, agent_id, agent_kwargs in tasks:
        dataset = make_dataset(dataset_id, device="cuda")
        agent = make_agent(agent_id, dataset, device="cuda", **agent_kwargs)

        task_id = _task_id(dataset.id, agent.id)
        agent_step = task_kwargs.get("agent_step", 0)
        if agent_step > 0:
            agent.load(task_id, agent_step)
        print(
            f"task={task_id}, dataset={dataset.id}, agent={agent.id}, "
            f"agent_kwargs={agent_kwargs}"
        )
        env = dataset.make_env(**env_kwargs)
        path = test(
            task_id,
            agent,
            env,
            episodes=task_kwargs.get("episodes", 100),
            seed=task_kwargs.get("seed", 42),
            print_interval=task_kwargs.get("print_interval", 1),
        )
        print(f"saved: {path}")

    view_test(tasks)


if __name__ == "__main__":
    main()
