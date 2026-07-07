from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train_model

DATASETS = [
    # "hopper_d4rl_medium",
    # "hopper_d4rl_hybrid",
    # "hopper_d4rl_expert",
    # "hopper_replay_medium",
    # "hopper_replay_expert",
    "halfcheetah_d4rl_medium",
    "halfcheetah_d4rl_hybrid",
    "halfcheetah_d4rl_expert",
    "halfcheetah_replay_medium",
    "halfcheetah_replay_expert",
]

AGENTS = [
    ([None, 50_000], "bc"),
    ([None, 100_000], "td3bc_n"),
    ([None, 200_000], "iql"),
    ([None, 500_000], "cql"),
    ([None, 500_000], "aspl_gp"),
    ([100_000, 500_000], "scas_gp"),
    ([100_000, 500_000], "scaspl_gp"),
]

TASKS = [
    # ([None, 500_000], "hopper_d4rl_medium", "cql", {"threshold": 1.5}),
    # ([None, 500_000], "hopper_d4rl_hybrid", "aspl", {"weight_punish": 0.5}),
    # ([100_000, 500_000], "hopper_d4rl_medium", "scaspl_gp", {"weight_punish": 1.0}),
]

INTERVAL = 1_000
COUNT = 10


def train_min_agent(
    task_start: list[int | None],
    dataset_id: str,
    agent_id: str,
    agent_kwargs: dict,
) -> None:
    model_start, agent_start = task_start

    dataset = make_dataset(dataset_id, device="cuda")
    agent = make_agent(agent_id, dataset, device="cuda", model_step=model_start, **agent_kwargs)
    task_id = _task_id(dataset.id, agent.id)

    if agent_start > 0:
        agent.load(task_id, agent_start)

    path = train_model(
        agent=agent,
        dataset=dataset,
        task_id=task_id,
        start=agent_start,
        steps=agent_start + INTERVAL * COUNT,
        save_interval=INTERVAL,
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    tasks = [
        (task_start, dataset_id, agent_id, {})
        for task_start, agent_id in AGENTS
        for dataset_id in DATASETS
    ] + TASKS

    for task_start, dataset_id, agent_id, agent_kwargs in tasks:
        train_min_agent(task_start, dataset_id, agent_id, agent_kwargs)
