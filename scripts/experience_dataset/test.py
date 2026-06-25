from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.test import test

DATASETS = [
    "hopper_d4rl_medium",
    "hopper_d4rl_hybrid",
    "hopper_d4rl_expert",
    "hopper_replay_medium",
    "hopper_replay_expert",
]

AGENTS = [
    (500_000, None, "bc_deterministic"),
    (500_000, None, "bc_stochastic"),
    (500_000, None, "td3bc"),
    (500_000, None, "iql"),
    (500_000, None, "cql"),
    (500_000, None, "aspl"),
    # (500_000, 100_000, "sdc"),
    # (500_000, 100_000, "sdc_cql"),
    (500_000, 100_000, "scas_lambda_0"),
    (500_000, 100_000, "scas_lambda_25"),
    (500_000, 100_000, "scas_lambda_50"),
    (500_000, 100_000, "scas_lambda_75"),
    (500_000, 100_000, "scas_lambda_100"),
    (500_000, 100_000, "scaspl"),
]


def test_agent(
    dataset_id: str,
    agent_step: int,
    model_step: int | None,
    agent_id: str,
) -> None:
    dataset = make_dataset(dataset_id, device="cuda")
    agent = make_agent(
        agent_id,
        dataset,
        device="cuda",
        model_step=model_step,
    )

    task_id = _task_id(dataset.id, agent.id)
    if agent_step > 0:
        agent.load(task_id, agent_step)
    print(
        f"task={task_id}, dataset={dataset.id}, agent={agent.id}, "
        f"agent_step={agent_step}, model_step={model_step}"
    )
    env = dataset.make_env()
    path = test(
        task_id,
        agent,
        env,
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    for agent_step, model_step, agent_id in AGENTS:
        for dataset_id in DATASETS:
            test_agent(dataset_id, agent_step, model_step, agent_id)
