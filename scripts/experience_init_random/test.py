from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.test import test


DATASETS = [
    ("hopper_d4rl_medium_noise_1", "hopper_d4rl_medium", {"reset_noise_scale": 5e-4}),
    ("hopper_d4rl_medium_noise_2", "hopper_d4rl_medium", {"reset_noise_scale": 5e-3}),
    ("hopper_d4rl_medium_noise_3", "hopper_d4rl_medium", {"reset_noise_scale": 5e-2}),
    ("hopper_d4rl_medium_noise_4", "hopper_d4rl_medium", {"reset_noise_scale": 5e-1}),
    ("hopper_d4rl_expert_noise_1", "hopper_d4rl_expert", {"reset_noise_scale": 5e-4}),
    ("hopper_d4rl_expert_noise_2", "hopper_d4rl_expert", {"reset_noise_scale": 5e-3}),
    ("hopper_d4rl_expert_noise_3", "hopper_d4rl_expert", {"reset_noise_scale": 5e-2}),
    ("hopper_d4rl_expert_noise_4", "hopper_d4rl_expert", {"reset_noise_scale": 5e-1}),
    ("hopper_replay_medium_noise_1", "hopper_replay_medium", {"reset_noise_scale": 5e-4}),
    ("hopper_replay_medium_noise_2", "hopper_replay_medium", {"reset_noise_scale": 5e-3}),
    ("hopper_replay_medium_noise_3", "hopper_replay_medium", {"reset_noise_scale": 5e-2}),
    ("hopper_replay_medium_noise_4", "hopper_replay_medium", {"reset_noise_scale": 5e-1}),
    ("hopper_replay_expert_noise_1", "hopper_replay_expert", {"reset_noise_scale": 5e-4}),
    ("hopper_replay_expert_noise_2", "hopper_replay_expert", {"reset_noise_scale": 5e-3}),
    ("hopper_replay_expert_noise_3", "hopper_replay_expert", {"reset_noise_scale": 5e-2}),
    ("hopper_replay_expert_noise_4", "hopper_replay_expert", {"reset_noise_scale": 5e-1}),
]

DATASET_SOURCES = {dataset_id: source_dataset_id for dataset_id, source_dataset_id, _ in DATASETS}

AGENTS = [
    ([500_000, 0], "bc_deterministic", {}),
    ([500_000, 0], "bc_stochastic", {}),
    ([500_000, 0], "td3bc", {}),
    ([500_000, 0], "iql", {}),
    ([500_000, 0], "cql", {}),
    ([500_000, 0], "aspl", {}),
    ([500_000, 100_000], "sdc", {}),
    ([500_000, 100_000], "sdc_cql", {}),
    ([500_000, 100_000], "scas", {}),
    ([500_000, 100_000], "scas_aspl", {}),
]


TASKS = [
    (task_steps, {}, dataset_id, env_kwargs, agent_id, agent_kwargs)
    for task_steps, agent_id, agent_kwargs in AGENTS
    for dataset_id, _, env_kwargs in DATASETS
]


def test_agent(
    task_steps: list[int],
    test_kwargs: dict,
    dataset_id: str,
    env_kwargs: dict,
    agent_id: str,
    agent_kwargs: dict,
) -> None:
    agent_step, model_step = task_steps
    source_dataset_id = DATASET_SOURCES[dataset_id]
    dataset = make_dataset(source_dataset_id, device="cuda")
    agent = make_agent(agent_id, dataset, device="cuda", model_step=model_step, **agent_kwargs)

    train_id = _task_id(source_dataset_id, agent.id)
    agent.load(train_id, agent_step)

    env = dataset.make_env(**env_kwargs)
    task_id = _task_id(dataset_id, agent.id)
    print(
        f"task={task_id}, train={train_id}, dataset={dataset_id}, source_dataset={source_dataset_id}, "
        f"agent={agent.id}, reset_noise_scale={env_kwargs['reset_noise_scale']:g}"
    )
    path = test(task_id, agent, env, **test_kwargs)
    print(f"saved: {path}")


if __name__ == "__main__":
    for task_steps, test_kwargs, dataset_id, env_kwargs, agent_id, agent_kwargs in TASKS:
        test_agent(task_steps, test_kwargs, dataset_id, env_kwargs, agent_id, agent_kwargs)
