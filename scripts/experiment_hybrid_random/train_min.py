from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train_model

DATASETS = [
    "walker2d_random_expert_1",
    "walker2d_random_expert_3",
    "walker2d_random_expert_5",
    "walker2d_random_expert_7",
    "walker2d_random_expert_9",
]

AGENTS = [
    ([None, 50_000], "bc"),
    ([None, 100_000], "td3bc_n"),
    ([None, 200_000], "iql"),
    # ([None, 500_000], "aspl"),
    # ([None, 500_000], "cql"),
    # ([100_000, 500_000], "scas_gp"),
    ([100_000, 500_000], "scaspl_gp"),
]

TASKS = [
    # ([None, 500_000], "walker2d_random_expert_7", "cql", {"threshold": 1.0}),
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
