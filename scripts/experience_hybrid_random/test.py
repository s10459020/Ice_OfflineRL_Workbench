from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.eval import cal_main
from ice_offline.run.test import test
from view import save_boxplots
from view import save_tables

DATASETS = [
    "walker2d_random_expert_1",
    "walker2d_random_expert_3",
    "walker2d_random_expert_5",
    "walker2d_random_expert_7",
    "walker2d_random_expert_9",
]

AGENTS = [
    (500_000, 100_000, "scaspl"),
]


def test_agent(
    dataset_id: str,
    agent_step: int,
    model_step: int | None,
    agent_id: str,
) -> None:
    dataset = make_dataset(dataset_id, device="cuda")
    agent = make_agent(
        agent_id,
        dataset,
        device="cuda",
        model_step=model_step,
    )

    task_id = _task_id(dataset.id, agent.id)
    if agent_step > 0:
        agent.load(task_id, agent_step)

    print(
        f"task={task_id}, dataset={dataset.id}, agent={agent.id}, "
        f"agent_step={agent_step}, model_step={model_step}"
    )
    env = dataset.make_env()
    path = test(task_id, agent, env)
    print(f"saved: {path}")


if __name__ == "__main__":
    agent_ids = [agent_id for _, _, agent_id in AGENTS]
    tasks = [
        (dataset_id, agent_step, model_step, agent_id)
        for agent_step, model_step, agent_id in AGENTS
        for dataset_id in DATASETS
    ]

    for dataset_id, agent_step, model_step, agent_id in tasks:
        test_agent(dataset_id, agent_step, model_step, agent_id)
        returns_output_path, _ = cal_main(_task_id(dataset_id, agent_id))
        print(f"saved: {returns_output_path}")

    save_tables(DATASETS, agent_ids)
    save_boxplots(DATASETS, agent_ids)
