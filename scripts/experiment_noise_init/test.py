from pathlib import Path

from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import eval_path
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import returns_path
from ice_offline.config.paths import task_id
from ice_offline.dataset._lookup import make_dataset
from plot import analyze
from plot import plot
from ice_offline.run.test import run
from ice_offline.store.eval.collector import EvalCollector
from view import save_boxplots
from view import save_tables

EXPERIMENT = "noise_init"

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

AGENTS = [
    ("bc", None, 50_000),
    ("td3bc_n", None, 100_000),
    ("iql", None, 200_000),
    ("cql", None, 500_000),
    ("aspl_gp", None, 500_000),
    ("scas_gp", 100_000, 500_000),
    ("scaspl_gp", 100_000, 500_000),
]

COUNT = 20
EVALS = 100
INTERVAL = 1_000


def _steps(start_step: int) -> list[int]:
    return [start_step + INTERVAL * index for index in range(COUNT + 1)]


def test(
    task_id: str,
    train_dataset_id: str,
    env_kwargs: dict,
    agent_id: str,
    model_step: int | None,
    agent_steps: list[int],
) -> Path:
    dataset = make_dataset(train_dataset_id, device="cuda")
    train_id = task_id(train_dataset_id, agent_id)
    path = eval_path(task_id)
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
        eval_col.save(path)
    finally:
        eval_col.close()

    print(f"saved: {path}")
    return path


if __name__ == "__main__":
    for test_dataset_id, train_dataset_id, env_kwargs in DATASETS:
        for agent_id, model_step, agent_step in AGENTS:
            id = experiment_task_id(EXPERIMENT, agent_id, test_dataset_id)
            path = test(
                id,
                train_dataset_id,
                env_kwargs,
                agent_id,
                model_step,
                _steps(agent_step),
            )
            analyze(id, path)
            plot(id, returns_path(id), test_dataset_id, agent_id)

    dataset_ids = [dataset_id for dataset_id, _, _ in DATASETS]
    agent_ids = [agent_id for agent_id, _, _ in AGENTS]
    save_tables(dataset_ids, agent_ids)
    save_boxplots(dataset_ids, agent_ids)
