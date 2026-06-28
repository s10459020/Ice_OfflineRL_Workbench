import gymnasium as gym
import numpy as np

from ice_offline.agent._spec import Agent
from ice_offline.config.paths import main_data_path
from ice_offline.store.state._lookup import STATE_OPS
from ice_offline.store.minari.collector import MinariCollectorWrapper
from ice_offline.tools.printer import print_stage


def run(agent: Agent, env: gym.Env, seed: int = 42) -> float:
    agent.set_seed(seed)
    o, _ = env.reset(seed=seed)
    result = 0.0
    trun = term = False
    while not (trun or term):
        a = agent.act(o)
        o, r, trun, term, _ = env.step(a)
        result += float(r)
    return result


def _state_io(env: gym.Env):
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


def run_noise_dynamic(
    agent: Agent,
    env: gym.Env,
    *,
    scale_noise: float = 5e-3,
    seed: int = 42,
) -> float:
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
    agent: Agent,
    env: gym.Env,
    *,
    episodes: int = 100,
    print_interval: int = 1,
    seed: int = 42,
) -> object:
    path = main_data_path("test", task_id)
    minari_col = MinariCollectorWrapper(env)

    print_stage(f"Test {task_id}")
    for i in range(episodes):
        result = run(agent, minari_col, seed + i)
        if (i + 1) % print_interval == 0:
            print(f"test episode={i + 1}/{episodes} return={result:.6g}")

    minari_col.save(path, id=task_id, agent_id=agent.id)
    minari_col.close()
    return path


def test_noise_dynamic(
    task_id: str,
    agent: Agent,
    env: gym.Env,
    *,
    scale_noise: float = 5e-3,
    episodes: int = 100,
    print_interval: int = 1,
    seed: int = 42,
) -> object:
    path = main_data_path("test", task_id)
    minari_col = MinariCollectorWrapper(env)

    print_stage(f"Test {task_id} noise_dynamic={scale_noise:g}")
    for i in range(episodes):
        result = run_noise_dynamic(agent, minari_col, scale_noise=scale_noise, seed=seed + i)
        if (i + 1) % print_interval == 0:
            print(f"test episode={i + 1}/{episodes} return={result:.6g}")

    minari_col.save(path, id=task_id, agent_id=agent.id)
    minari_col.close()
    return path


if __name__ == "__main__":
    from ice_offline.agent._lookup import make_agent
    from ice_offline.dataset._lookup import make_dataset

    device = "cuda:0"
    task_id = "check_run-v0"
    dataset = make_dataset("hopper_simple", device=device)
    agent = make_agent("td3bc", dataset, device=device)
    agent.load(task_id, 20_000)
    env = dataset.make_env()
    path = test(task_id, agent, env, episodes=10)
    print(path)
