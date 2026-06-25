from ice_offline.config.paths import _task_id
from ice_offline.config.paths import table_path
from ice_offline.run.eval import cal_dataset
from ice_offline.run.eval import cal_main
from ice_offline.run.table import table_mean
from ice_offline.run.table import table_pr95
from ice_offline.run.table import table_true

DATASETS = [
    ("hopper_d4rl_medium_noise_1", "hopper_random", "hopper_d4rl_medium"),
    ("hopper_d4rl_medium_noise_2", "hopper_random", "hopper_d4rl_medium"),
    ("hopper_d4rl_medium_noise_3", "hopper_random", "hopper_d4rl_medium"),
    ("hopper_d4rl_medium_noise_4", "hopper_random", "hopper_d4rl_medium"),
    ("hopper_d4rl_expert_noise_1", "hopper_random", "hopper_d4rl_expert"),
    ("hopper_d4rl_expert_noise_2", "hopper_random", "hopper_d4rl_expert"),
    ("hopper_d4rl_expert_noise_3", "hopper_random", "hopper_d4rl_expert"),
    ("hopper_d4rl_expert_noise_4", "hopper_random", "hopper_d4rl_expert"),
    ("hopper_replay_medium_noise_1", "hopper_random", "hopper_replay_medium"),
    ("hopper_replay_medium_noise_2", "hopper_random", "hopper_replay_medium"),
    ("hopper_replay_medium_noise_3", "hopper_random", "hopper_replay_medium"),
    ("hopper_replay_medium_noise_4", "hopper_random", "hopper_replay_medium"),
    ("hopper_replay_expert_noise_1", "hopper_random", "hopper_replay_expert"),
    ("hopper_replay_expert_noise_2", "hopper_random", "hopper_replay_expert"),
    ("hopper_replay_expert_noise_3", "hopper_random", "hopper_replay_expert"),
    ("hopper_replay_expert_noise_4", "hopper_random", "hopper_replay_expert"),
]

AGENTS = [
    "bc_deterministic",
    "bc_stochastic",
    "td3bc",
    "iql",
    "cql",
    "aspl",
    "sdc",
    "sdc_cql",
    "scas",
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


if __name__ == "__main__":
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

    table_true(dataset_ids, AGENTS, datas, lowers, uppers, table_path("experience_init_random", "true_returns.csv"))
    table_mean(dataset_ids, AGENTS, datas, lowers, uppers, table_path("experience_init_random", "mean_returns.csv"))
    table_pr95(dataset_ids, AGENTS, datas, lowers, uppers, table_path("experience_init_random", "pr95_returns.csv"))
