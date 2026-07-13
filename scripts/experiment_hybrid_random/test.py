from pathlib import Path

from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import eval_data_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.test import run
from ice_offline.store.eval.collector import EvalCollector
from plot import eval
from plot import plot
from train_min import INTERVAL
from view import TABLES
from view import save_boxplots
from view import save_table_boxplot
from view import save_tables

DATASETS = [
    "walker2d_random_expert_1",
    "walker2d_random_expert_3",
    "walker2d_random_expert_5",
    "walker2d_random_expert_7",
    "walker2d_random_expert_9",
]

AGENTS = [
    ("scaspl", 100_000, 500_000),
]

EVALS = 100
COUNT = 20


def _steps(start_step: int) -> list[int]:
    return [start_step + INTERVAL * index for index in range(COUNT + 1)]


def test(
    task_id: str,
    dataset_id: str,
    agent_id: str,
    model_step: int | None,
    agent_steps: list[int],
) -> Path:
    dataset = make_dataset(dataset_id, device="cuda")
    eval_path = eval_data_path("test", task_id)
    eval_col = EvalCollector(dataset.make_env())
    try:
        for agent_step in agent_steps:
            agent = make_agent(agent_id, dataset, device="cuda", model_step=model_step)
            agent.load(task_id, agent_step)
            print(f"task={task_id}, agent_step={agent_step}, model_step={model_step}")
            for index in range(EVALS):
                result = run(agent, eval_col, 42 + index)
                print(f"test step={agent_step} episode={index + 1}/{EVALS} return={result:.6g}")
            eval_col.flush(agent_step)
        eval_col.save(eval_path)
    finally:
        eval_col.close()

    print(f"saved: {eval_path}")
    return eval_path


if __name__ == "__main__":
    for dataset_id in DATASETS:
        for agent_id, model_step, agent_step in AGENTS:
            task_id = _task_id(dataset_id, agent_id)
            path = test(task_id, dataset_id, agent_id, model_step, _steps(agent_step))
            returns_rows = eval(task_id, path)
            plot(task_id, returns_rows)

    agent_ids = [agent_id for agent_id, _, _ in AGENTS]
    save_tables(DATASETS, agent_ids)
    save_table_boxplot(TABLES)
    save_boxplots(DATASETS, agent_ids)
