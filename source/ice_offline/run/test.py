import gymnasium as gym

from ice_offline.agent._spec import Agent
from ice_offline.config.paths import data_path
from ice_offline.store.eval.collector import EvalCollector
from ice_offline.tools.printer import print_stage

def set_seed(agent, env, seed):
    agent.set_seed(seed)
    o, _ = env.reset(seed=seed)
    return o

def eval(agent: Agent, env: gym.Env, seed: int = 42, count: int = 1) -> list[float]:
    returns: list[float] = []
    for i in range(count):
        o = set_seed(agent, env, seed + i)
        result = 0.0
        trun = term = False
        while not (trun or term):
            a = agent.act(o)
            o, r, trun, term, _ = env.step(a)
            result += float(r)
        returns.append(result)
    return returns

def test(
    task_id: str,
    agent: Agent,
    env: gym.Env,
    *,
    episodes: int = 1,
    print_interval = 1,
    mode: str = "test",
    seed: int = 42,
) -> object:
    path = data_path(mode, task_id)
    eval_col = EvalCollector(env)

    print_stage(f"{mode.capitalize()} {task_id}")
    for i in range(1, episodes + 1):
        eval_seed = seed + i - 1
        eval(agent, eval_col, eval_seed)
        eval_col.flush(i)
        if i % print_interval == 0:
            print(f"{mode} episode={i}/{episodes}")

    eval_col.save(path)
    eval_col.close()
    return path
