from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.train import train
from config import AGENT_TASKS
from config import DATASETS
from plot import plot_agent


def train_agent(
    task_steps: list[int],
    dataset_id: str,
    agent_id: str,
    agent_kwargs: dict,
) -> None:
    agent_step, model_step = task_steps
    dataset = make_dataset(dataset_id, device="cuda")
    agent = make_agent(agent_id, dataset, device="cuda", model_step=model_step, **agent_kwargs)
    task_id = _task_id(dataset.id, agent.id)

    path = train(
        agent=agent,
        dataset=dataset,
        task_id=task_id,
        steps=agent_step,
        eval_env=dataset.make_eval_env(),
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    for task_steps, dataset_id, agent_id, agent_kwargs in AGENT_TASKS:
        train_agent(task_steps, dataset_id, agent_id, agent_kwargs)
        plot_agent(DATASETS.index(dataset_id) + 1, dataset_id, agent_id)
