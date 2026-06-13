from ice_offline.config.paths import _task_id
from ice_offline.config.paths import data_path_train
from ice_offline.config.paths import eval_returns_path
from ice_offline.config.paths import eval_steps_path
from ice_offline.run.eval import cal_returns
from ice_offline.run.eval import cal_steps


DATASET_ID_LIST = [
    # "hopper_random",
    # "hopper_replay",
    # "hopper_medium_replay",
    # "hopper_medium_d4rl",
    # "hopper_expert_d4rl",
    # "hopper_medium_expert",
    "hopper_simple",
    # "hopper_medium",
    # "hopper_expert",
    # "walker2d_random",
    # "walker2d_replay",
    # "walker2d_medium_replay",
    # "walker2d_medium_d4rl",
    # "walker2d_expert_d4rl",
    # "walker2d_medium_expert",
    # "walker2d_simple",
    # "walker2d_medium",
    # "walker2d_expert",
    # "halfcheetah_random",
    # "halfcheetah_replay",
    # "halfcheetah_medium_replay",
    # "halfcheetah_medium_d4rl",
    # "halfcheetah_expert_d4rl",
    # "halfcheetah_medium_expert",
    # "halfcheetah_simple",
    # "halfcheetah_medium",
    # "halfcheetah_expert",
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


def main() -> None:
    for dataset_id in DATASET_ID_LIST:
        for agent_id in AGENT_ID_LIST:
            task_id = _task_id(dataset_id, agent_id)
            input_path = data_path_train(task_id)
            if not input_path.exists():
                print(f"skip missing: {input_path}")
                continue

            returns_path = eval_returns_path(task_id)
            steps_path = eval_steps_path(task_id)
            print(f"evals={input_path}")
            cal_returns(input_path, returns_path)
            cal_steps(input_path, steps_path)
            print(f"saved: {returns_path}")
            print(f"saved: {steps_path}")


if __name__ == "__main__":
    main()
