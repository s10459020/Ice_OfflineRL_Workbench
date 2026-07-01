import shutil

from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import custom_dataset_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.test import test


DEVICE = "cuda"
SEED = 42

TASKS = [
    ((200_000, None, 1), "road_one", "hopper_d4rl_medium", "bc", {"reset_noise_scale": 0.0}),
    ((200_000, None, 1), "road_one", "hopper_d4rl_expert", "bc", {"reset_noise_scale": 0.0}),
]


def collect_dataset(
    task_steps: tuple[int, int | None, int],
    task_prefix: str,
    dataset_id: str,
    agent_id: str,
    env_kwargs: dict,
) -> None:
    agent_step, model_step, episodes = task_steps
    output_dataset_id = dataset_id.replace("_d4rl_", "_road_")

    dataset = make_dataset(dataset_id, device=DEVICE)
    agent = make_agent(agent_id, dataset, device=DEVICE, model_step=model_step)

    model_task_id = _task_id(dataset_id, agent.id)
    task_id = _task_id(f"{task_prefix}@{dataset_id}", agent.id)
    if agent_step > 0:
        agent.load(model_task_id, agent_step)

    print("====================================")
    print(f"task: {task_id}")
    print(f"env_kwargs: {env_kwargs}")
    print("====================================")

    env = dataset.make_env(**env_kwargs)
    path = test(
        task_id=task_id,
        agent=agent,
        env=env,
        episodes=episodes,
        seed=SEED,
    )
    env.close()
    print(f"saved: {path}")

    output_path = custom_dataset_path(output_dataset_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, output_path)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    for task_steps, task_prefix, dataset_id, agent_id, env_kwargs in TASKS:
        collect_dataset(task_steps, task_prefix, dataset_id, agent_id, env_kwargs)
