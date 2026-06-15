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

TEST_KWARGS = {
    "model_step": 100_000,
    "episodes": 100,
    "print_interval": 1,
}

DATASETS = [
    # {"run": "hopper_random", "lower": "hopper_random", "upper": 1000.0},
    # {"run": "hopper_replay_expert", "lower": "hopper_random", "upper": "hopper_replay_expert"},
    # {"run": "hopper_replay_medium", "lower": "hopper_random", "upper": "hopper_replay_medium"},
    # {"run": "hopper_d4rl_medium", "lower": "hopper_random", "upper": "hopper_d4rl_medium"},
    # {"run": "hopper_d4rl_hybrid", "lower": "hopper_random", "upper": "hopper_d4rl_hybrid"},
    # {"run": "hopper_d4rl_expert", "lower": "hopper_random", "upper": "hopper_d4rl_expert"},
    {"run": "hopper_simple", "lower": "hopper_random", "upper": "hopper_simple"},
    {"run": "hopper_medium", "lower": "hopper_random", "upper": "hopper_medium"},
    {"run": "hopper_expert", "lower": "hopper_random", "upper": "hopper_expert"},
    # {"run": "walker2d_random", "lower": None, "upper": None},
    # {"run": "walker2d_replay_expert", "lower": "walker2d_random", "upper": "walker2d_replay_expert"},
    # {"run": "walker2d_replay_medium", "lower": "walker2d_random", "upper": "walker2d_replay_medium"},
    # {"run": "walker2d_d4rl_medium", "lower": "walker2d_random", "upper": "walker2d_d4rl_medium"},
    # {"run": "walker2d_d4rl_hybrid", "lower": "walker2d_random", "upper": "walker2d_d4rl_hybrid"},
    # {"run": "walker2d_d4rl_expert", "lower": "walker2d_random", "upper": "walker2d_d4rl_expert"},
    # {"run": "walker2d_simple", "lower": "walker2d_random", "upper": "walker2d_simple"},
    # {"run": "walker2d_medium", "lower": "walker2d_random", "upper": "walker2d_medium"},
    # {"run": "walker2d_expert", "lower": "walker2d_random", "upper": "walker2d_expert"},
    # {"run": "halfcheetah_random", "lower": None, "upper": None},
    # {"run": "halfcheetah_replay_expert", "lower": "halfcheetah_random", "upper": "halfcheetah_replay_expert"},
    # {"run": "halfcheetah_replay_medium", "lower": "halfcheetah_random", "upper": "halfcheetah_replay_medium"},
    # {"run": "halfcheetah_d4rl_medium", "lower": "halfcheetah_random", "upper": "halfcheetah_d4rl_medium"},
    # {"run": "halfcheetah_d4rl_hybrid", "lower": "halfcheetah_random", "upper": "halfcheetah_d4rl_hybrid"},
    # {"run": "halfcheetah_d4rl_expert", "lower": "halfcheetah_random", "upper": "halfcheetah_d4rl_expert"},
    # {"run": "halfcheetah_simple", "lower": "halfcheetah_random", "upper": "halfcheetah_simple"},
    # {"run": "halfcheetah_medium", "lower": "halfcheetah_random", "upper": "halfcheetah_medium"},
    # {"run": "halfcheetah_expert", "lower": "halfcheetah_random", "upper": "halfcheetah_expert"},
]

AGENT_ID_LIST = [
    # "bc_deterministic",
    # "bc_stochastic",
    # "td3bc",
    # "iql",
    # "cql",
    # "cql_max_q",
    # "cql_soft_q",
    "aspl",
    # "sdc_cql",
    # "sdc_pre",
    # "scas_min",
    # "scas_mean",
    # "scas_aspl",
]

TABLE_OUTPUT_DIR = VIEW_ROOT / "table"
TEST_RETURNS: dict[str, list[float]] = {}


def view_test(datasets: list[dict[str, str | float | None]], agent_ids: list[str]) -> None:
    save_test_views(datasets, agent_ids)
    dataset_ids = [item["run"] for item in datasets]
    data_paths = [
        [returns_path("test", _task_id(item["run"], agent_id)) for agent_id in agent_ids]
        for item in datasets
    ]
    lower_paths = [_bound_path(item["lower"]) for item in datasets]
    upper_paths = [_bound_path(item["upper"]) for item in datasets]
    table_true(dataset_ids, agent_ids, data_paths, lower_paths, upper_paths, TABLE_OUTPUT_DIR / "true_returns.csv")
    table_mean(dataset_ids, agent_ids, data_paths, lower_paths, upper_paths, TABLE_OUTPUT_DIR / "mean_returns.csv")
    table_pr95(dataset_ids, agent_ids, data_paths, lower_paths, upper_paths, TABLE_OUTPUT_DIR / "pr95_returns.csv")


def save_test_views(datasets: list[dict[str, str | float | None]], agent_ids: list[str]) -> None:
    for item in datasets:
        dataset_id = item["run"]
        for agent_id in agent_ids:
            task_id = _task_id(dataset_id, agent_id)
            values = TEST_RETURNS[task_id]
            path = returns_path("test", task_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as file:
                json.dump(values, file)
            print(f"saved: {path}")


def save_dataset_views(datasets: list[dict[str, str | float | None]]) -> None:
    dataset_ids = []
    for item in datasets:
        for value in [item["run"], item["lower"], item["upper"]]:
            if isinstance(value, str) and value not in dataset_ids:
                dataset_ids.append(value)

    for dataset_id in dataset_ids:
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


def main() -> None:
    test_kwargs = {k: v for k, v in TEST_KWARGS.items() if v is not None}
    save_dataset_views(DATASETS)
    for item in DATASETS:
        dataset_id = item["run"]
        dataset = make_dataset(dataset_id, device="cuda")

        for agent_id in AGENT_ID_LIST:
            agent = make_agent(agent_id, dataset, device="cuda")
            task_id = _task_id(dataset.id, agent.id)
            model_step = test_kwargs.get("model_step", 0)
            if model_step > 0:
                agent.load(task_id, model_step)
            print(f"task={task_id}, dataset={dataset.id}, agent={agent.id}")
            returns = test(
                agent=agent,
                dataset=dataset,
                task_id=task_id,
                episodes=test_kwargs.get("episodes", 100),
                seed=test_kwargs.get("seed", 42),
                print_interval=test_kwargs.get("print_interval", 1),
            )
            TEST_RETURNS[task_id] = returns
            print(f"avg_returns={sum(returns) / len(returns):.2f}")
    view_test(DATASETS, AGENT_ID_LIST)


if __name__ == "__main__":
    main()
