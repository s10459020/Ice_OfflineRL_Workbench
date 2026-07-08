from pathlib import Path

from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import eval_data_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.dataset.eval import EvalDataset
from ice_offline.run.boxplot import write_boxplots
from ice_offline.run.eval import EvalRows
from ice_offline.run.eval import eval_returns
from ice_offline.run.eval import eval_steps
from ice_offline.run.eval import write_eval_rows
from ice_offline.run.table import write_tables
from ice_offline.run.test import run
from ice_offline.store.eval.collector import EvalCollector
from train_min import COUNT
from train_min import INTERVAL

DATASETS = [
    "walker2d_random_expert_1",
    "walker2d_random_expert_3",
    "walker2d_random_expert_5",
    "walker2d_random_expert_7",
    "walker2d_random_expert_9",
]

TABLES = [
    ("walker2d_random_expert_1", "walker2d_random", "walker2d_d4rl_expert"),
    ("walker2d_random_expert_3", "walker2d_random", "walker2d_d4rl_expert"),
    ("walker2d_random_expert_5", "walker2d_random", "walker2d_d4rl_expert"),
    ("walker2d_random_expert_7", "walker2d_random", "walker2d_d4rl_expert"),
    ("walker2d_random_expert_9", "walker2d_random", "walker2d_d4rl_expert"),
]

AGENTS = [
    ("scaspl", 100_000, 500_000),
]

EVALS = 100
VALUE_CACHE: dict[str, list[float]] = {}


def _steps(start_step: int) -> list[int]:
    return [start_step + INTERVAL * index for index in range(COUNT + 1)]


def _value(dataset_id: str) -> list[float]:
    if dataset_id in VALUE_CACHE:
        return VALUE_CACHE[dataset_id]
    dataset = make_dataset(dataset_id, device="cpu")
    values = [
        float(episode.rewards.sum())
        for episode in dataset.episodes
    ]
    VALUE_CACHE[dataset_id] = values
    return values


def _cache(task_id: str, rows: EvalRows) -> None:
    VALUE_CACHE[task_id] = [
        value
        for _, values in rows
        for value in values
    ]


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


def eval(task_id: str, eval_path: Path) -> EvalRows:
    eval_dataset = EvalDataset(path=eval_path, device="cpu")
    batches = eval_dataset.batch_episodes
    returns_rows = eval_returns(batches)
    steps_rows = eval_steps(batches)
    returns_output_path, steps_output_path = write_eval_rows("test", task_id, returns_rows, steps_rows)
    print(f"saved: {returns_output_path}")
    print(f"saved: {steps_output_path}")
    return returns_rows


if __name__ == "__main__":
    for dataset_id in DATASETS:
        for agent_id, model_step, agent_step in AGENTS:
            task_id = _task_id(dataset_id, agent_id)
            path = test(task_id, dataset_id, agent_id, model_step, _steps(agent_step))
            returns_rows = eval(task_id, path)
            _cache(task_id, returns_rows)

    dataset_ids, lower_ids, upper_ids = map(list, zip(*TABLES))
    agent_ids = [agent_id for agent_id, _, _ in AGENTS]
    data_values = [
        [VALUE_CACHE.get(_task_id(dataset_id, agent_id)) for agent_id in agent_ids]
        for dataset_id in dataset_ids
    ]
    lower_values = [_value(lower_id) for lower_id in lower_ids]
    upper_values = [_value(upper_id) for upper_id in upper_ids]
    write_tables(
        "experience_hybrid_random",
        dataset_ids,
        agent_ids,
        data_values,
        lower_values,
        upper_values,
    )
    write_boxplots(
        "experience_hybrid_random",
        dataset_ids,
        agent_ids,
        data_values,
        lower_values,
        upper_values,
    )
