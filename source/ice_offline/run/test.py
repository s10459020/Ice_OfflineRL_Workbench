import gymnasium as gym
from pathlib import Path

from ice_offline.agent._spec import Agent
from ice_offline.config.paths import eval_path
from ice_offline.config.paths import model_path
from ice_offline.store.eval.collector import EvalCollector
from ice_offline.tools.printer import print_stage


def _run(agent: Agent, env: gym.Env, seed: int = 42) -> float:
    agent.set_seed(seed)
    o, _ = env.reset(seed=seed)
    result = 0.0
    trun = term = False
    while not (trun or term):
        a = agent.act(o)
        o, r, trun, term, _ = env.step(a)
        result += float(r)
    return result


def test_eval(
    task_id: str,
    train_id: str,
    agent: Agent,
    env: gym.Env,
    steps: list[int],
    *,
    episodes: int = 100,
    print_interval: int = 1,
    seed: int = 42,
) -> Path:
    path = eval_path(task_id)
    eval_col = EvalCollector(env, resume_path=path)
    try:
        for step in steps:
            agent.load(model_path(train_id, step))
            print_stage(f"Test {task_id} step={step}")
            for i in range(episodes):
                result = _run(agent, eval_col, seed + i)
                if (i + 1) % print_interval == 0:
                    print(f"test step={step} episode={i + 1}/{episodes} return={result:.6g}")
            eval_col.flush(step)
        eval_col.save(path)
    finally:
        eval_col.close()
    return path


if __name__ == "__main__":
    from ice_offline.agent._lookup import make_agent
    from ice_offline.dataset._lookup import make_dataset

    device = "cuda:0"
    task_id = "check_run-v0"
    dataset = make_dataset("hopper_simple", device=device)
    agent = make_agent("td3bc", dataset, device=device)
    env = dataset.make_env()
    path = test_eval(task_id, task_id, agent, env, [20_000], episodes=10)
    print(path)
