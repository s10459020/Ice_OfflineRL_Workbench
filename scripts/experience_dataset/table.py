from ice_offline.config.paths import _task_id
from ice_offline.config.paths import table_path
from ice_offline.run.eval import cal_dataset
from ice_offline.run.eval import cal_main
from ice_offline.run.table import table_mean
from ice_offline.run.table import table_pr95
from ice_offline.run.table import table_true

DATASETS = [
    ("hopper_d4rl_medium", "hopper_random", "hopper_d4rl_medium"),
    ("hopper_d4rl_hybrid", "hopper_random", "hopper_d4rl_hybrid"),
    ("hopper_d4rl_expert", "hopper_random", "hopper_d4rl_expert"),
    ("hopper_replay_medium", "hopper_random", "hopper_d4rl_medium"),
    ("hopper_replay_expert", "hopper_random", "hopper_d4rl_expert"),
    ("halfcheetah_d4rl_medium", "halfcheetah_random", "halfcheetah_d4rl_medium"),
    ("halfcheetah_d4rl_hybrid", "halfcheetah_random", "halfcheetah_d4rl_hybrid"),
    ("halfcheetah_d4rl_expert", "halfcheetah_random", "halfcheetah_d4rl_expert"),
    ("halfcheetah_replay_medium", "halfcheetah_random", "halfcheetah_d4rl_medium"),
    ("halfcheetah_replay_expert", "halfcheetah_random", "halfcheetah_d4rl_expert"),
]

AGENTS = [
    "bc",
    "td3bc",
    "iql",
    "cql",
    "aspl",
    # "sdc",
    # "sdc_cql",
    "scas",
    # "scas_lambda_0",
    # "scas_lambda_25",
    # "scas_lambda_50",
    # "scas_lambda_75",
    # "scas_lambda_100",
    "scaspl",
]


def save_test_view(dataset_id: str, agent_id: str):
    task_id = _task_id(dataset_id, agent_id)
    result = cal_main(task_id)
    if result is None:
        return None
     
    returns_output_path, _ = result
    print(f"saved: {returns_output_path}")
    return returns_output_path


def build_tables() -> None:
    dataset_ids = [dataset_id for dataset_id, _, _ in DATASETS]
    datas: list[list[object]] = []
    lowers: list[object] = []
    uppers: list[object] = []
    bounds: dict[str, object] = {}

    for dataset_id, lower_id, upper_id in DATASETS:
        datas.append([
            save_test_view(dataset_id, agent_id)
            for agent_id in AGENTS
        ])
        if lower_id not in bounds:
            bounds[lower_id] = cal_dataset(lower_id)[0]
        if upper_id not in bounds:
            bounds[upper_id] = cal_dataset(upper_id)[0]
        lowers.append(bounds[lower_id])
        uppers.append(bounds[upper_id])

    table_true(dataset_ids, AGENTS, datas, lowers, uppers, table_path("experience_dataset", "true_returns.csv"))
    table_mean(dataset_ids, AGENTS, datas, lowers, uppers, table_path("experience_dataset", "mean_returns.csv"))
    table_pr95(dataset_ids, AGENTS, datas, lowers, uppers, table_path("experience_dataset", "pr95_returns.csv"))


if __name__ == "__main__":
    build_tables()
