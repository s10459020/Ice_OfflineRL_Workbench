from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.test import test_noise_dynamic
from table import build_tables

DATASETS = [
    ("hopper_d4rl_medium_noise_dynamic", "hopper_d4rl_medium"),
    ("hopper_d4rl_expert_noise_dynamic", "hopper_d4rl_expert"),
    ("hopper_replay_medium_noise_dynamic", "hopper_replay_medium"),
    ("hopper_replay_expert_noise_dynamic", "hopper_replay_expert"),
]

AGENTS = [
    (500_000, 0, "bc"),
    (500_000, 0, "td3bc"),
    (500_000, 0, "iql"),
    (500_000, 0, "cql"),
    (500_000, 0, "aspl"),
    (500_000, 100_000, "scas"),
    (500_000, 100_000, "scaspl"),
]


def test_agent(
    test_dataset_id: str,
    train_dataset_id: str,
    agent_id: str,
    agent_step: int,
    model_step: int,
) -> None:
    dataset = make_dataset(train_dataset_id, device="cuda")
    agent = make_agent(agent_id, dataset, device="cuda", model_step=model_step)

    train_id = _task_id(train_dataset_id, agent.id)
    agent.load(train_id, agent_step)

    task_id = _task_id(test_dataset_id, agent.id)
    env = dataset.make_env()
    print("====================================")
    print(f"task: {task_id}")
    print("dynamic_noise_scale: 5e-3")
    print("====================================")

    path = test_noise_dynamic(task_id, agent, env, scale_noise=5e-3)
    print(f"saved: {path}")


if __name__ == "__main__":
    tasks = [
        (test_dataset_id, train_dataset_id, agent_id, agent_step, model_step)
        for agent_step, model_step, agent_id in AGENTS
        for test_dataset_id, train_dataset_id in DATASETS
    ]

    for test_dataset_id, train_dataset_id, agent_id, agent_step, model_step in tasks:
        test_agent(test_dataset_id, train_dataset_id, agent_id, agent_step, model_step)
    build_tables()
