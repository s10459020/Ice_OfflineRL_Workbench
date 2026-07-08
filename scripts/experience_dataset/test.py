from pathlib import Path

from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import eval_data_path
from ice_offline.dataset._lookup import make_dataset
from ice_offline.run.test import run
from ice_offline.store.eval.collector import EvalCollector
from view import ensure_agent_eval
from view import save_boxplots
from view import save_tables

DATASETS = [
    # "hopper_d4rl_medium",
    # "hopper_d4rl_hybrid",
    # "hopper_d4rl_expert",
    # "hopper_replay_medium",
    # "hopper_replay_expert",
    "walker2d_d4rl_medium",
    "walker2d_d4rl_hybrid",
    "walker2d_d4rl_expert",
    "walker2d_replay_medium",
    "walker2d_replay_expert",
    # "halfcheetah_d4rl_medium",
    # "halfcheetah_d4rl_hybrid",
    # "halfcheetah_d4rl_expert",
    # "halfcheetah_replay_medium",
    # "halfcheetah_replay_expert",
]

AGENTS = [
    # ("bc", None, 50_000),
    # ("td3bc_n", None, 100_000),
    # ("iql", None, 200_000),
    # ("cql", None, 500_000),
    # ("scas_gp", 100_000, 500_000),
    # ("scaspl_gp", 100_000, 500_000),
    ("aspl_gp_punish_005", None, 500_000),
    ("aspl_gp_punish_010", None, 500_000),
    ("aspl_gp_punish_050", None, 500_000),
]

COUNT = 10
EVALS = 100
INTERVAL = 1_000


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
            print(f"task={task_id}, agent_step={agent_step}")
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
            agent_steps = _steps(agent_step)
            test(task_id, dataset_id, agent_id, model_step, agent_steps)
            ensure_agent_eval(dataset_id, agent_id)

    agent_ids = [agent_id for agent_id, _, _ in AGENTS]
    save_tables(DATASETS, agent_ids)
    save_boxplots(DATASETS, agent_ids)
