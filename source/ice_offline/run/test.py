import gymnasium as gym

from ice_offline.agent._spec import Agent
from ice_offline.config.paths import data_path
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


def test(
    task_id: str,
    agent: Agent,
    env: gym.Env,
    *,
    episodes: int = 100,
    print_interval: int = 1,
    seed: int = 42,
) -> object:
    path = data_path("test", task_id)
    minari_col = MinariCollectorWrapper(env)

    print_stage(f"Test {task_id}")
    for i in range(episodes):
        result = run(agent, minari_col, seed + i)
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
    agent = make_agent("bc_stochastic", dataset, device=device)
    agent.load(task_id, 20_000)
    env = dataset.make_env()
    path = test(task_id, agent, env, episodes=10)
    print(path)
