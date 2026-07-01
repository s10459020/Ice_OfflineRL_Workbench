from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train


DATASETS = [
    "hopper_replay_medium",
    "hopper_replay_expert",
]

TASK_STEPS = [None, 0, 200_000]

AGENTS = [
    ("cql_gap_0p0", "cql", {"threshold": 0.0}),
    ("cql_gap_0p5", "cql", {"threshold": 0.5}),
    ("cql_gap_1p0", "cql", {"threshold": 1.0}),
    ("cql_gap_1p5", "cql", {"threshold": 1.5}),
    ("cql_gp_gap_0p0", "cql_gp", {"threshold": 0.0, "weight_gp": 1.0, "gp_threshold": 1.0, "gp_count": 16}),
    ("cql_gp_gap_0p5", "cql_gp", {"threshold": 0.5, "weight_gp": 1.0, "gp_threshold": 1.0, "gp_count": 16}),
    ("cql_gp_gap_1p0", "cql_gp", {"threshold": 1.0, "weight_gp": 1.0, "gp_threshold": 1.0, "gp_count": 16}),
    ("cql_gp_gap_1p5", "cql_gp", {"threshold": 1.5, "weight_gp": 1.0, "gp_threshold": 1.0, "gp_count": 16}),
]


def train_agent(
    task_steps: list[int | None],
    dataset_id: str,
    task_agent_id: str,
    agent_id: str,
    agent_kwargs: dict[str, object],
) -> None:
    model_start, agent_start, steps = task_steps
    dataset = make_dataset(dataset_id, device="cuda")
    config = dict(agent_kwargs)
    agent = make_agent(agent_id, dataset, device="cuda", model_step=model_start, **config)
    task_id = _task_id(dataset.id, task_agent_id)

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
        (TASK_STEPS, dataset_id, task_agent_id, agent_id, agent_kwargs)
        for dataset_id in DATASETS
        for task_agent_id, agent_id, agent_kwargs in AGENTS
    ]

    for task_steps, dataset_id, task_agent_id, agent_id, agent_kwargs in tasks:
        train_agent(task_steps, dataset_id, task_agent_id, agent_id, agent_kwargs)
