from ice_offline.config.paths import _task_id
from ice_offline.run.train import train
from _config import make_agent
from _config import make_dataset


TRAIN_KWARGS = {
    "start": 0,
    "steps": 500_000,
    # "save_interval": 20_000,
    # "eval_interval": 2_000,
    # "print_interval": 200,
    # "eval_episodes": 20,
}


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
    "td3bc",
    # "iql",
    # "cql",
    # "cql_max_q",
    # "cql_soft_q",
    # "aspl",
    # "sdc_cql",
    # "sdc_pre",
    # "scas_min",
    # "scas_mean",
    # "scas_aspl",
]


def main() -> None:
    train_kwargs = {key: value for key, value in TRAIN_KWARGS.items() if value is not None}
    start = train_kwargs.get("start", 0)
    for dataset_id in DATASET_ID_LIST:
        dataset = make_dataset(dataset_id)

        for agent_id in AGENT_ID_LIST:
            agent = make_agent(agent_id, dataset)
            
            task_id = _task_id(dataset.id, agent.id)
            if start > 0:
                agent.load(task_id, start)
            print(f"task={task_id}, dataset={dataset.id}, agent={agent.id}")
            path = train(
                agent=agent,
                dataset=dataset,
                task_id=task_id,
                **train_kwargs,
            )
            print(f"saved: {path}")


if __name__ == "__main__":
    main()
