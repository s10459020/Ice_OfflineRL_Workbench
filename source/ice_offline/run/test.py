import gymnasium as gym

from ice_offline.agent._spec import Agent
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import data_path_test
from ice_offline.dataset.base import Dataset
from ice_offline.store.eval.collector import EvalCollector
from ice_offline.tools.printer import print_stage


EPISODES = 10
PRINT_INTERVAL = 1
SEED = 42


def eval(
    agent: Agent,
    env: gym.Env,
    episodes: int = EPISODES,
    seed: int = SEED,
) -> list[float]:
    returns: list[float] = []
    for i in range(1, episodes + 1):
        now_seed = seed + i
        agent.set_seed(now_seed)
        o, _ = env.reset(seed=now_seed)

        result = 0.0
        trun = term = False
        while not (trun or term):
            a = agent.act(o)
            o, r, trun, term, _ = env.step(a)
            result += float(r)

        returns.append(result)

    return returns


def test(
    agent: Agent,
    dataset: Dataset,
    *,
    task_id: str | None = None,
    episodes: int = EPISODES,
    seed: int = SEED,
    print_interval: int = PRINT_INTERVAL,
    env_kwargs: dict | None = None,
) -> object:
    task_id = task_id or _task_id(dataset.id, agent.id)
    env = dataset.make_env(**(env_kwargs or {}))

    path = data_path_test(task_id)
    eval_col = EvalCollector(env)

    print_stage(f"Test {agent.id} in {dataset.id}")
    for episode in range(1, episodes + 1):
        eval(agent, eval_col, 1, seed + episode - 1)
        eval_col.flush(episode)
        if print_interval > 0 and episode % print_interval == 0:
            print(f"test episode={episode}/{episodes}")
            
    eval_col.save(path)
    eval_col.close()
    return path
