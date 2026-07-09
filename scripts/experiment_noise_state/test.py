from pathlib import Path

import numpy as np

from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import eval_data_path
from ice_offline.dataset._lookup import make_dataset
from plot import eval
from plot import plot
from ice_offline.store.eval.collector import EvalCollector
from ice_offline.store.state._lookup import STATE_OPS
from view import save_boxplots
from view import save_tables

DATASETS = [
    ("noise_state_5e-4@hopper_d4rl_medium", "hopper_d4rl_medium", 5e-4),
    ("noise_state_5e-3@hopper_d4rl_medium", "hopper_d4rl_medium", 5e-3),
    ("noise_state_5e-2@hopper_d4rl_medium", "hopper_d4rl_medium", 5e-2),
    ("noise_state_5e-1@hopper_d4rl_medium", "hopper_d4rl_medium", 5e-1),
    ("noise_state_5e-4@hopper_d4rl_hybrid", "hopper_d4rl_hybrid", 5e-4),
    ("noise_state_5e-3@hopper_d4rl_hybrid", "hopper_d4rl_hybrid", 5e-3),
    ("noise_state_5e-2@hopper_d4rl_hybrid", "hopper_d4rl_hybrid", 5e-2),
    ("noise_state_5e-1@hopper_d4rl_hybrid", "hopper_d4rl_hybrid", 5e-1),
    ("noise_state_5e-4@hopper_replay_medium", "hopper_replay_medium", 5e-4),
    ("noise_state_5e-3@hopper_replay_medium", "hopper_replay_medium", 5e-3),
    ("noise_state_5e-2@hopper_replay_medium", "hopper_replay_medium", 5e-2),
    ("noise_state_5e-1@hopper_replay_medium", "hopper_replay_medium", 5e-1),
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


def _steps(start_step: int) -> list[int]:
    return [start_step + INTERVAL * index for index in range(COUNT + 1)]


def _state_io(env):
    env_id = env.unwrapped.spec.id
    _, state_io_cls, _ = STATE_OPS[env_id]
    return state_io_cls(env)


def _noise_state(state, scale_noise: float):
    payload = state.serialize()
    noisy_payload = {}
    for key, value in payload.items():
        value_np = np.asarray(value, dtype=np.float64)
        noisy_payload[key] = value_np + scale_noise * np.random.randn(*value_np.shape)
    return state.__class__.from_serialized(noisy_payload)


def run_noise_state(agent, env, *, scale_noise: float = 5e-3, seed: int = 42) -> float:
    agent.set_seed(seed)
    np.random.seed(seed)
    state_io = _state_io(env)
    o, _ = env.reset(seed=seed)
    result = 0.0
    trun = term = False
    while not (trun or term):
        a = agent.act(o)
        o, r, trun, term, _ = env.step(a)
        result += float(r)
        if trun or term:
            continue
        state = state_io.get_state()
        state_noisy = _noise_state(state, scale_noise)
        state_io.set_state(state_noisy)
        o = env.unwrapped._get_obs()
    return result


def test(
    task_id: str,
    train_dataset_id: str,
    scale_noise: float,
    agent_id: str,
    model_step: int | None,
    agent_steps: list[int],
) -> Path:
    dataset = make_dataset(train_dataset_id, device="cuda")
    train_id = _task_id(train_dataset_id, agent_id)
    eval_path = eval_data_path("test", task_id)
    eval_col = EvalCollector(dataset.make_env())
    try:
        for agent_step in agent_steps:
            agent = make_agent(agent_id, dataset, device="cuda", model_step=model_step)
            agent.load(train_id, agent_step)
            print(f"task={task_id}, train_id={train_id}, agent_step={agent_step}, state_noise_scale={scale_noise:g}")
            for index in range(EVALS):
                result = run_noise_state(
                    agent,
                    eval_col,
                    scale_noise=scale_noise,
                    seed=42 + index,
                )
                print(f"test step={agent_step} episode={index + 1}/{EVALS} return={result:.6g}")
            eval_col.flush(agent_step)
        eval_col.save(eval_path)
    finally:
        eval_col.close()

    print(f"saved: {eval_path}")
    return eval_path


if __name__ == "__main__":
    for test_dataset_id, train_dataset_id, scale_noise in DATASETS:
        for agent_id, model_step, agent_step in AGENTS:
            task_id = _task_id(test_dataset_id, agent_id)
            path = test(
                task_id,
                train_dataset_id,
                scale_noise,
                agent_id,
                model_step,
                _steps(agent_step),
            )
            returns_rows = eval(task_id, path)
            plot(task_id, returns_rows)

    dataset_ids = [dataset_id for dataset_id, _, _ in DATASETS]
    agent_ids = [agent_id for agent_id, _, _ in AGENTS]
    save_tables(dataset_ids, agent_ids)
    save_boxplots(dataset_ids, agent_ids)
