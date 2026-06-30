import numpy as np

from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.eval import cal_main
from ice_offline.run.test import test
from view import save_boxplots
from view import save_tables

DATASETS = [
    ("noise_action_5e-3@hopper_d4rl_medium", "hopper_d4rl_medium", 5e-3),
    ("noise_action_5e-3@hopper_d4rl_expert", "hopper_d4rl_expert", 5e-3),
    ("noise_action_5e-3@hopper_replay_medium", "hopper_replay_medium", 5e-3),
    ("noise_action_5e-3@hopper_replay_expert", "hopper_replay_expert", 5e-3),
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


def _noise_action(action, scale_noise: float):
    action_np = np.asarray(action, dtype=np.float32)
    return action_np + scale_noise * np.random.randn(*action_np.shape)


def run_noise_action(agent, env, *, scale_noise: float = 5e-3, seed: int = 42) -> float:
    agent.set_seed(seed)
    np.random.seed(seed)
    o, _ = env.reset(seed=seed)
    result = 0.0
    trun = term = False
    while not (trun or term):
        a = agent.act(o)
        a = _noise_action(a, scale_noise)
        o, r, trun, term, _ = env.step(a)
        result += float(r)
    return result


def test_agent(
    test_dataset_id: str,
    train_dataset_id: str,
    scale_noise: float,
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
    print(f"dataset: {train_dataset_id}")
    print(f"action_noise_scale: {scale_noise:g}")
    print("====================================")

    path = test(
        task_id,
        agent,
        env,
        run_callback=lambda agent, env, seed: run_noise_action(
            agent,
            env,
            scale_noise=scale_noise,
            seed=seed,
        ),
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    agent_ids = [agent_id for _, _, agent_id in AGENTS]
    tasks = [
        (test_dataset_id, train_dataset_id, scale_noise, agent_id, agent_step, model_step)
        for agent_step, model_step, agent_id in AGENTS
        for test_dataset_id, train_dataset_id, scale_noise in DATASETS
    ]

    for test_dataset_id, train_dataset_id, scale_noise, agent_id, agent_step, model_step in tasks:
        test_agent(test_dataset_id, train_dataset_id, scale_noise, agent_id, agent_step, model_step)
        returns_output_path, _ = cal_main(_task_id(test_dataset_id, agent_id))
        print(f"saved: {returns_output_path}")

    dataset_ids = [dataset_id for dataset_id, _, _ in DATASETS]
    save_tables(dataset_ids, agent_ids)
    save_boxplots(dataset_ids, agent_ids)
