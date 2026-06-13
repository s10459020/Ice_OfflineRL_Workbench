from ice_offline.config.paths import _task_id
from ice_offline.config.paths import eval_returns_path
from ice_offline.config.paths import eval_steps_path
from ice_offline.config.paths import metric_path
from ice_offline.config.paths import plot_path
from ice_offline.run.plot import plot


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
    for index, dataset_id in enumerate(DATASET_ID_LIST, start=1):
        for agent_id in AGENT_ID_LIST:
            task_id = _task_id(dataset_id, agent_id)
            metric_paths = [metric_path(task_id)]
            eval_paths = [
                eval_returns_path(task_id),
                eval_steps_path(task_id),
            ]
            output_path = plot_path(index, dataset_id, agent_id)

            paths = metric_paths + eval_paths
            missing_paths = [path for path in paths if not path.exists()]
            if missing_paths:
                for path in missing_paths:
                    print(f"skip missing: {path}")
                continue

            print(f"plot dataset={dataset_id}, agent={agent_id}")
            plot(metric_paths, eval_paths, output_path)
            print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
