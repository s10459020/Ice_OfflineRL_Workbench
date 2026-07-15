import numpy as np

from ice_offline.agent._lookup import make_agent
from ice_offline.config.paths import eval_path
from ice_offline.config.paths import experiment_task_id
from ice_offline.config.paths import returns_path
from ice_offline.dataset._lookup import make_dataset
from plot import analyze
from plot import plot
from ice_offline.run.test import test_eval
from ice_offline.store.state._lookup import STATE_OPS
from view import save_boxplots
from view import save_tables

EXPERIMENT = "noise_state"
EXPERIMENT_TRAIN = "base_train"

DATASETS = [
    ("noise_state_5e-4@walker2d_d4rl_medium", "walker2d_d4rl_medium", 5e-4),
    ("noise_state_1e-3@walker2d_d4rl_medium", "walker2d_d4rl_medium", 1e-3),
    ("noise_state_5e-3@walker2d_d4rl_medium", "walker2d_d4rl_medium", 5e-3),
    ("noise_state_1e-2@walker2d_d4rl_medium", "walker2d_d4rl_medium", 1e-2),
    ("noise_state_5e-4@walker2d_d4rl_hybrid", "walker2d_d4rl_hybrid", 5e-4),
    ("noise_state_1e-3@walker2d_d4rl_hybrid", "walker2d_d4rl_hybrid", 1e-3),
    ("noise_state_5e-3@walker2d_d4rl_hybrid", "walker2d_d4rl_hybrid", 5e-3),
    ("noise_state_1e-2@walker2d_d4rl_hybrid", "walker2d_d4rl_hybrid", 1e-2),
    ("noise_state_5e-4@walker2d_replay_medium", "walker2d_replay_medium", 5e-4),
    ("noise_state_1e-3@walker2d_replay_medium", "walker2d_replay_medium", 1e-3),
    ("noise_state_5e-3@walker2d_replay_medium", "walker2d_replay_medium", 5e-3),
    ("noise_state_1e-2@walker2d_replay_medium", "walker2d_replay_medium", 1e-2),
]

AGENTS = [
    # ("bc", None, 50_000),
    ("td3bc_n", None, 100_000),
    # ("iql", None, 200_000),
    # ("cql", None, 500_000),
    # ("aspl_gp", None, 500_000),
    # ("scas_gp", 100_000, 500_000),
    ("scas_gp", 100_000, 500_000),
    ("scaspl_n", 100_000, 500_000),
]

COUNT = 20
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
    test_dataset_id: str,
    train_dataset_id: str,
    scale_noise: float,
    agent_id: str,
    model_step: int | None,
    start_step: int,
) -> str:
    test_id = experiment_task_id(EXPERIMENT, agent_id, test_dataset_id)
    train_id = experiment_task_id(EXPERIMENT_TRAIN, agent_id, train_dataset_id)
    model_train_id = experiment_task_id(EXPERIMENT_TRAIN, "scas_model", train_dataset_id)
    steps = _steps(start_step)

    dataset = make_dataset(train_dataset_id, device="cuda")
    agent = make_agent(agent_id, dataset, device="cuda", model_step=model_step, model_train_id=model_train_id)

    def runner(agent, env, seed: int) -> float:
        return run_noise_state(agent, env, scale_noise=scale_noise, seed=seed)

    print(f"task={test_id}, train_id={train_id}, state_noise_scale={scale_noise:g}")
    path = test_eval(
        test_id,
        train_id,
        agent,
        dataset.make_env(),
        steps,
        episodes=EVALS,
        runner=runner,
    )
    print(f"saved: {path}")
    return test_id


if __name__ == "__main__":
    for agent_id, model_step, agent_step in AGENTS:
        for test_dataset_id, train_dataset_id, scale_noise in DATASETS:
            test_id = test(
                test_dataset_id,
                train_dataset_id,
                scale_noise,
                agent_id,
                model_step,
                agent_step,
            )
            analyze(test_id, eval_path(test_id))
            plot(test_id, returns_path(test_id), test_dataset_id, agent_id)

    dataset_ids = [dataset_id for dataset_id, _, _ in DATASETS]
    agent_ids = [agent_id for agent_id, _, _ in AGENTS]
    save_tables(dataset_ids, agent_ids)
    save_boxplots(dataset_ids, agent_ids)
