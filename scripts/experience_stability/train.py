from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train
from plot import plot_agent

DATASETS = [
    "walker2d_d4rl_medium",
    "walker2d_d4rl_hybrid",
    "walker2d_d4rl_expert",
    "walker2d_replay_medium",
    "walker2d_replay_expert",
]

AGENTS = [
    # ([None, 0, 50_000], "td3"),
    # ([None, 0, 50_000], "td3_n"),
    # ([None, 0, 50_000], "td3_gp"),
    # ([None, 0, 50_000], "td3_gpn"),
    # ([None, 0, 100_000], "td3bc"),
    # ([None, 0, 100_000], "td3bc_n"),
    # ([None, 0, 100_000], "td3bc_gp"),
    # ([None, 0, 100_000], "td3bc_gpn"),
    # ([None, 0, 500_000], "cql"),
    ([None, 0, 500_000], "cql_threshold_n5"),
    # ([None, 0, 500_000], "cql_threshold_5"),
    # ([None, 0, 500_000], "cql_gp"),
    # ([100_000, 0, 500_000], "scas"),
    # ([100_000, 0, 500_000], "scas_n"),
    # ([100_000, 0, 500_000], "scas_gp"),
    # ([100_000, 0, 500_000], "scas_gpn"),
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
        eval_env=dataset.make_eval_env(),
        steps=steps,
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    tasks = [
        (task_steps, dataset_id, agent_id, {})
        for task_steps, agent_id in AGENTS
        for dataset_id in DATASETS
    ]

    for task_steps, dataset_id, agent_id, agent_kwargs in tasks:
        train_agent(task_steps, dataset_id, agent_id, agent_kwargs)
        plot_agent(dataset_id, agent_id)
