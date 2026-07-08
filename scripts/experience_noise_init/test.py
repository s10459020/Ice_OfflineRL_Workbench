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

DATASETS = [
    ("noise_init_5e-4@hopper_d4rl_medium", "hopper_d4rl_medium", {"reset_noise_scale": 5e-4}),
    ("noise_init_5e-3@hopper_d4rl_medium", "hopper_d4rl_medium", {"reset_noise_scale": 5e-3}),
    ("noise_init_5e-2@hopper_d4rl_medium", "hopper_d4rl_medium", {"reset_noise_scale": 5e-2}),
    ("noise_init_5e-1@hopper_d4rl_medium", "hopper_d4rl_medium", {"reset_noise_scale": 5e-1}),
    ("noise_init_5e-4@hopper_d4rl_hybrid", "hopper_d4rl_hybrid", {"reset_noise_scale": 5e-4}),
    ("noise_init_5e-3@hopper_d4rl_hybrid", "hopper_d4rl_hybrid", {"reset_noise_scale": 5e-3}),
    ("noise_init_5e-2@hopper_d4rl_hybrid", "hopper_d4rl_hybrid", {"reset_noise_scale": 5e-2}),
    ("noise_init_5e-1@hopper_d4rl_hybrid", "hopper_d4rl_hybrid", {"reset_noise_scale": 5e-1}),
    ("noise_init_5e-4@hopper_replay_medium", "hopper_replay_medium", {"reset_noise_scale": 5e-4}),
    ("noise_init_5e-3@hopper_replay_medium", "hopper_replay_medium", {"reset_noise_scale": 5e-3}),
    ("noise_init_5e-2@hopper_replay_medium", "hopper_replay_medium", {"reset_noise_scale": 5e-2}),
    ("noise_init_5e-1@hopper_replay_medium", "hopper_replay_medium", {"reset_noise_scale": 5e-1}),
]

TABLES = [
    ("noise_init_5e-4@hopper_d4rl_medium", "hopper_random", "hopper_d4rl_medium"),
    ("noise_init_5e-3@hopper_d4rl_medium", "hopper_random", "hopper_d4rl_medium"),
    ("noise_init_5e-2@hopper_d4rl_medium", "hopper_random", "hopper_d4rl_medium"),
    ("noise_init_5e-1@hopper_d4rl_medium", "hopper_random", "hopper_d4rl_medium"),
    ("noise_init_5e-4@hopper_d4rl_hybrid", "hopper_random", "hopper_d4rl_hybrid"),
    ("noise_init_5e-3@hopper_d4rl_hybrid", "hopper_random", "hopper_d4rl_hybrid"),
    ("noise_init_5e-2@hopper_d4rl_hybrid", "hopper_random", "hopper_d4rl_hybrid"),
    ("noise_init_5e-1@hopper_d4rl_hybrid", "hopper_random", "hopper_d4rl_hybrid"),
    ("noise_init_5e-4@hopper_replay_medium", "hopper_random", "hopper_replay_medium"),
    ("noise_init_5e-3@hopper_replay_medium", "hopper_random", "hopper_replay_medium"),
    ("noise_init_5e-2@hopper_replay_medium", "hopper_random", "hopper_replay_medium"),
    ("noise_init_5e-1@hopper_replay_medium", "hopper_random", "hopper_replay_medium"),
]

AGENTS = [
    ("bc", None, 50_000),
    ("td3bc_n", None, 100_000),
    ("iql", None, 200_000),
    ("cql", None, 500_000),
    ("aspl_gp", None, 500_000),
    ("scas_gp", 100_000, 500_000),
    ("scaspl_gp", 100_000, 500_000),
]

COUNT = 10
EVALS = 100
INTERVAL = 1_000
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
    train_dataset_id: str,
    env_kwargs: dict,
    agent_id: str,
    model_step: int | None,
    agent_steps: list[int],
) -> Path:
    dataset = make_dataset(train_dataset_id, device="cuda")
    train_id = _task_id(train_dataset_id, agent_id)
    eval_path = eval_data_path("test", task_id)
    eval_col = EvalCollector(dataset.make_env(**env_kwargs))
    try:
        for agent_step in agent_steps:
            agent = make_agent(agent_id, dataset, device="cuda", model_step=model_step)
            agent.load(train_id, agent_step)
            print(
                f"task={task_id}, train_id={train_id}, agent_step={agent_step}, "
                f"reset_noise_scale={env_kwargs['reset_noise_scale']:g}"
            )
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
    for test_dataset_id, train_dataset_id, env_kwargs in DATASETS:
        for agent_id, model_step, agent_step in AGENTS:
            task_id = _task_id(test_dataset_id, agent_id)
            path = test(
                task_id,
                train_dataset_id,
                env_kwargs,
                agent_id,
                model_step,
                _steps(agent_step),
            )
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
        "experience_noise_init",
        dataset_ids,
        agent_ids,
        data_values,
        lower_values,
        upper_values,
    )
    write_boxplots(
        "experience_noise_init",
        dataset_ids,
        agent_ids,
        data_values,
        lower_values,
        upper_values,
    )
