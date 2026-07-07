from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train
from plot import plot_agent

DATASETS = [
    # "walker2d_random_expert_1",
    "walker2d_random_expert_3",
    "walker2d_random_expert_5",
    "walker2d_random_expert_7",
    "walker2d_random_expert_9",
]

AGENTS = [
    # ([None, 0, 50_000], "bc"),
    # ([None, 0, 100_000], "td3bc_n"),
    # ([None, 0, 200_000], "iql"),
    # ([None, 0, 500_000], "aspl"),
    # ([None, 0, 500_000], "cql"),
    # ([100_000, 0, 500_000], "scas_gp"),
    ([100_000, 0, 500_000], "scaspl_gp"),
]

TASKS = [
    # ([None, 0, 500_000], dataset_id, "cql", {"threshold": 1.0})
    # ([None, 0, 500_000], dataset_id, "aspl", {"weight_punish": 1.0})
    # ([100_000, 0, 500_000], dataset_id, "scas", {})
    # *(([100_000, 0, 500_000], dataset_id, "scaspl", {"weight_punish": 1.0, "weight_correction": 0.25}) for dataset_id in DATASETS),
]


def train_agent(
    task_steps: list[int | None],
    dataset_id: str,
    agent_id: str,
    agent_kwargs: dict,
) -> None:
    model_start, agent_start, steps = task_steps
    dataset = make_dataset(dataset_id, device="cuda")
    agent = make_agent(agent_id, dataset, device="cuda", model_step=model_start, **agent_kwargs)
    task_id = _task_id(dataset.id, agent.id)

    if agent_start > 0:
        agent.load(task_id, agent_start)

    path = train(
        agent=agent,
        dataset=dataset,
        task_id=task_id,
        start=agent_start,
        eval_env=dataset.make_eval_env(),
        steps=steps,
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    tasks = [
        (task_steps, dataset_id, agent_id, {})
        for task_steps, agent_id in AGENTS
        for dataset_id in DATASETS
    ] + TASKS

    for task_steps, dataset_id, agent_id, agent_kwargs in tasks:
        train_agent(task_steps, dataset_id, agent_id, agent_kwargs)
        plot_agent(dataset_id, agent_id)
