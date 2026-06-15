from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.test import test

TEST_KWARGS = {
    "model_step": 100_000,
    "episodes": 100,
    "print_interval": 1,
}

DATASET_ID_LIST = [
    "hopper_random",
    "hopper_replay_expert",
    "hopper_replay_medium",
    "hopper_d4rl_medium",
    "hopper_d4rl_hybrid",
    "hopper_d4rl_expert",
    "hopper_simple",
    "hopper_medium",
    "hopper_expert",
    # "walker2d_random",
    # "walker2d_replay_expert",
    # "walker2d_replay_medium",
    # "walker2d_d4rl_medium",
    # "walker2d_d4rl_hybrid",
    # "walker2d_d4rl_expert",
    # "walker2d_simple",
    # "walker2d_medium",
    # "walker2d_expert",
    # "halfcheetah_random",
    # "halfcheetah_replay_expert",
    # "halfcheetah_replay_medium",
    # "halfcheetah_d4rl_medium",
    # "halfcheetah_d4rl_hybrid",
    # "halfcheetah_d4rl_expert",
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
    # "aspl",
    # "sdc_cql",
    # "sdc_pre",
    "scas_min",
    # "scas_mean",
    # "scas_aspl",
]


def main() -> None:
    test_kwargs = {k: v for k, v in TEST_KWARGS.items() if v is not None}
    for dataset_id in DATASET_ID_LIST:
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
            print(f"avg_returns={sum(returns) / len(returns):.2f}")


if __name__ == "__main__":
    main()
