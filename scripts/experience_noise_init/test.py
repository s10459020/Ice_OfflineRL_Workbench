from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.eval import cal_main
from ice_offline.run.test import test
from view import save_boxplots
from view import save_tables

DATASETS = [
    ("noise_init_5e-4@hopper_d4rl_medium", "hopper_d4rl_medium", {"reset_noise_scale": 5e-4}),
    ("noise_init_5e-3@hopper_d4rl_medium", "hopper_d4rl_medium", {"reset_noise_scale": 5e-3}),
    ("noise_init_5e-2@hopper_d4rl_medium", "hopper_d4rl_medium", {"reset_noise_scale": 5e-2}),
    ("noise_init_5e-1@hopper_d4rl_medium", "hopper_d4rl_medium", {"reset_noise_scale": 5e-1}),
    ("noise_init_5e-4@hopper_d4rl_expert", "hopper_d4rl_expert", {"reset_noise_scale": 5e-4}),
    ("noise_init_5e-3@hopper_d4rl_expert", "hopper_d4rl_expert", {"reset_noise_scale": 5e-3}),
    ("noise_init_5e-2@hopper_d4rl_expert", "hopper_d4rl_expert", {"reset_noise_scale": 5e-2}),
    ("noise_init_5e-1@hopper_d4rl_expert", "hopper_d4rl_expert", {"reset_noise_scale": 5e-1}),
    # ("noise_init_5e-4@hopper_replay_medium", "hopper_replay_medium", {"reset_noise_scale": 5e-4}),
    # ("noise_init_5e-3@hopper_replay_medium", "hopper_replay_medium", {"reset_noise_scale": 5e-3}),
    # ("noise_init_5e-2@hopper_replay_medium", "hopper_replay_medium", {"reset_noise_scale": 5e-2}),
    # ("noise_init_5e-1@hopper_replay_medium", "hopper_replay_medium", {"reset_noise_scale": 5e-1}),
    # ("noise_init_5e-4@hopper_replay_expert", "hopper_replay_expert", {"reset_noise_scale": 5e-4}),
    # ("noise_init_5e-3@hopper_replay_expert", "hopper_replay_expert", {"reset_noise_scale": 5e-3}),
    # ("noise_init_5e-2@hopper_replay_expert", "hopper_replay_expert", {"reset_noise_scale": 5e-2}),
    # ("noise_init_5e-1@hopper_replay_expert", "hopper_replay_expert", {"reset_noise_scale": 5e-1}),
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
    env_kwargs: dict,
    agent_id: str,
    agent_step: int,
    model_step: int,
) -> None:
    dataset = make_dataset(train_dataset_id, device="cuda")
    agent = make_agent(agent_id, dataset, device="cuda", model_step=model_step)

    train_id = _task_id(train_dataset_id, agent.id)
    agent.load(train_id, agent_step)

    task_id = _task_id(test_dataset_id, agent.id)
    env = dataset.make_env(**env_kwargs)
    print("====================================")
    print(f"task: {task_id}")
    print(f"reset_noise_scale: {env_kwargs['reset_noise_scale']:g}")
    print("====================================")

    path = test(task_id, agent, env)
    print(f"saved: {path}")


if __name__ == "__main__":
    agent_ids = [agent_id for _, _, agent_id in AGENTS]
    tasks = [
        (test_dataset_id, train_dataset_id, env_kwargs, agent_id, agent_step, model_step)
        for agent_step, model_step, agent_id in AGENTS
        for test_dataset_id, train_dataset_id, env_kwargs in DATASETS
    ]

    for test_dataset_id, train_dataset_id, env_kwargs, agent_id, agent_step, model_step in tasks:
        test_agent(test_dataset_id, train_dataset_id, env_kwargs, agent_id, agent_step, model_step)
        returns_output_path, _ = cal_main(_task_id(test_dataset_id, agent_id))
        print(f"saved: {returns_output_path}")

    dataset_ids = [dataset_id for dataset_id, _, _ in DATASETS]
    save_tables(dataset_ids, agent_ids)
    save_boxplots(dataset_ids, agent_ids)
