import json

from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import VIEW_ROOT
from ice_offline.config.paths import returns_path
from ice_offline.config.paths import steps_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.dataset import eval_dataset
from ice_offline.run.table import table_mean
from ice_offline.run.table import table_pr95
from ice_offline.run.table import table_true
from ice_offline.run.test import test

TABLES = {
    # "hopper_random": {"lower": "hopper_random", "upper": 1000.0},
    # "hopper_replay_expert": {"lower": "hopper_random", "upper": "hopper_replay_expert"},
    # "hopper_replay_medium": {"lower": "hopper_random", "upper": "hopper_replay_medium"},
    # "hopper_d4rl_medium": {"lower": "hopper_random", "upper": "hopper_d4rl_medium"},
    # "hopper_d4rl_hybrid": {"lower": "hopper_random", "upper": "hopper_d4rl_hybrid"},
    # "hopper_d4rl_expert": {"lower": "hopper_random", "upper": "hopper_d4rl_expert"},
    "hopper_simple": {"lower": "hopper_random", "upper": "hopper_simple"},
    "hopper_medium": {"lower": "hopper_random", "upper": "hopper_medium"},
    "hopper_expert": {"lower": "hopper_random", "upper": "hopper_expert"},
    # "walker2d_random": {"lower": None, "upper": None},
    # "walker2d_replay_expert": {"lower": "walker2d_random", "upper": "walker2d_replay_expert"},
    # "walker2d_replay_medium": {"lower": "walker2d_random", "upper": "walker2d_replay_medium"},
    # "walker2d_d4rl_medium": {"lower": "walker2d_random", "upper": "walker2d_d4rl_medium"},
    # "walker2d_d4rl_hybrid": {"lower": "walker2d_random", "upper": "walker2d_d4rl_hybrid"},
    # "walker2d_d4rl_expert": {"lower": "walker2d_random", "upper": "walker2d_d4rl_expert"},
    # "walker2d_simple": {"lower": "walker2d_random", "upper": "walker2d_simple"},
    # "walker2d_medium": {"lower": "walker2d_random", "upper": "walker2d_medium"},
    # "walker2d_expert": {"lower": "walker2d_random", "upper": "walker2d_expert"},
    # "halfcheetah_random": {"lower": None, "upper": None},
    # "halfcheetah_replay_expert": {"lower": "halfcheetah_random", "upper": "halfcheetah_replay_expert"},
    # "halfcheetah_replay_medium": {"lower": "halfcheetah_random", "upper": "halfcheetah_replay_medium"},
    # "halfcheetah_d4rl_medium": {"lower": "halfcheetah_random", "upper": "halfcheetah_d4rl_medium"},
    # "halfcheetah_d4rl_hybrid": {"lower": "halfcheetah_random", "upper": "halfcheetah_d4rl_hybrid"},
    # "halfcheetah_d4rl_expert": {"lower": "halfcheetah_random", "upper": "halfcheetah_d4rl_expert"},
    # "halfcheetah_simple": {"lower": "halfcheetah_random", "upper": "halfcheetah_simple"},
    # "halfcheetah_medium": {"lower": "halfcheetah_random", "upper": "halfcheetah_medium"},
    # "halfcheetah_expert": {"lower": "halfcheetah_random", "upper": "halfcheetah_expert"},
}

TASKS = [
    # ({"model_step": 200_000}, "hopper_simple", {}, "aspl", {"alpha": 0.5}),
    # ({"model_step": 200_000}, "hopper_medium", {}, "aspl", {"alpha": 0.5}),
    # ({"model_step": 200_000}, "hopper_expert", {}, "aspl", {"alpha": 0.5}),
    # ({"model_step": 200_000}, "hopper_simple", {}, "bc_deterministic", {}),
    # ({"model_step": 200_000}, "hopper_medium", {}, "bc_deterministic", {}),
    # ({"model_step": 200_000}, "hopper_expert", {}, "bc_deterministic", {}),
    # ({"model_step": 200_000}, "hopper_simple", {}, "td3bc", {}),
    # ({"model_step": 200_000}, "hopper_medium", {}, "td3bc", {}),
    # ({"model_step": 200_000}, "hopper_expert", {}, "td3bc", {}),
    # ({"model_step": 500_000}, "hopper_simple", {}, "cql_soft_q", {"threshold": 1.5}),
    # ({"model_step": 500_000}, "hopper_medium", {}, "cql_soft_q", {"threshold": 1.0}),
    # ({"model_step": 500_000}, "hopper_expert", {}, "cql_soft_q", {"threshold": 0.5}),
]

TASK_KWARGS = {
    "episodes": 100,
    "print_interval": 1,
}

DATASETS = [
    # ("hopper_simple", {}),
    # ("hopper_medium", {}),
    # ("hopper_expert", {}),
]


AGENTS = [
    # ("aspl", {"alpha": 0.5}),
    # ("bc_deterministic", {}),
    # ("td3bc", {}),
    # ("cql_soft_q", {"threshold": 1.5}),
]


TABLE_OUTPUT_DIR = VIEW_ROOT / "table"
TEST_RETURNS: dict[str, list[float]] = {}


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
    for _, dataset_id, _, agent_id, _ in tasks:
        task_id = _task_id(dataset_id, agent_id)
        values = TEST_RETURNS[task_id]
        path = returns_path("test", task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(values, file)
        print(f"saved: {path}")


def save_dataset_views(dataset_ids: list[str]) -> None:
    all_dataset_ids: list[str] = []
    for dataset_id in dataset_ids:
        if dataset_id not in all_dataset_ids:
            all_dataset_ids.append(dataset_id)

        for value in [TABLES[dataset_id]["lower"], TABLES[dataset_id]["upper"]]:
            if isinstance(value, str) and value not in all_dataset_ids:
                all_dataset_ids.append(value)

    for dataset_id in all_dataset_ids:
        returns_output_path = returns_path("dataset", dataset_id)
        steps_output_path = steps_path("dataset", dataset_id)
        if returns_output_path.exists() and steps_output_path.exists():
            continue
        dataset = make_dataset(dataset_id, device="cuda")
        eval_dataset(dataset, returns_output_path, steps_output_path)


def _bound_path(value: str | float | None):
    if value is None or isinstance(value, float):
        return value
    return returns_path("dataset", value)


def view_test(tasks: list[tuple[dict[str, object], str, dict[str, object], str, dict[str, object]]]) -> None:
    save_test_views(tasks)
    dataset_ids = TABLES.keys()
    agent_ids = [agent_id for _, _, _, agent_id, _ in tasks]
    data_path_rows: list[list[object]] = []
    for dataset_id in dataset_ids:
        row: list[object] = [None for _ in agent_ids]
        for _, task_dataset_id, _, agent_id, _ in tasks:
            if task_dataset_id != dataset_id:
                continue
            agent_index = agent_ids.index(agent_id)
            row[agent_index] = returns_path("test", _task_id(dataset_id, agent_id))
        data_path_rows.append(row)
    lower_paths = [_bound_path(TABLES[dataset_id]["lower"]) for dataset_id in dataset_ids]
    upper_paths = [_bound_path(TABLES[dataset_id]["upper"]) for dataset_id in dataset_ids]
    table_true(dataset_ids, agent_ids, data_path_rows, lower_paths, upper_paths, TABLE_OUTPUT_DIR / "true_returns.csv")
    table_mean(dataset_ids, agent_ids, data_path_rows, lower_paths, upper_paths, TABLE_OUTPUT_DIR / "mean_returns.csv")
    table_pr95(dataset_ids, agent_ids, data_path_rows, lower_paths, upper_paths, TABLE_OUTPUT_DIR / "pr95_returns.csv")


def main() -> None:
    tasks = normalize_tasks()
    dataset_ids = [dataset_id for _, dataset_id, _, _, _ in tasks]
    save_dataset_views(dataset_ids)

    for task_kwargs, dataset_id, env_kwargs, agent_id, agent_kwargs in tasks:
        dataset = make_dataset(dataset_id, device="cuda")
        agent = make_agent(agent_id, dataset, device="cuda", **agent_kwargs)

        task_id = _task_id(dataset.id, agent.id)
        model_step = task_kwargs.get("model_step", 0)
        if model_step > 0:
            agent.load(task_id, model_step)
        print(
            f"task={task_id}, dataset={dataset.id}, agent={agent.id}, "
            f"agent_kwargs={agent_kwargs}"
        )
        returns = test(
            agent=agent,
            dataset=dataset,
            task_id=task_id,
            episodes=task_kwargs.get("episodes", 100),
            seed=task_kwargs.get("seed", 42),
            print_interval=task_kwargs.get("print_interval", 1),
            env_kwargs=env_kwargs,
        )
        TEST_RETURNS[task_id] = returns
        print(f"avg_returns={sum(returns) / len(returns):.2f}")

    view_test(tasks)


if __name__ == "__main__":
    main()
