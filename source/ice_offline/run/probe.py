import gymnasium as gym

from ice_offline.agent._spec import Agent
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import data_path
from ice_offline.dataset.base import Dataset
from ice_offline.store.eval.collector import EvalCollector
from ice_offline.store.probe.op_collector import ProbeCollectWrapper
from ice_offline.store.probe.op_collector import ProbeEvalFn
from ice_offline.store.probe.op_collector import ProbeInterface
from ice_offline.tools.printer import print_stage
SEED = 42


def replay(
    dataset: Dataset,
    env: gym.Env,
    *,
    episodes: int | None = None,
    seed: int = SEED,
    print_interval: int = 1,
) -> None:
    episodes = dataset.episode_count if episodes is None else episodes
    print_stage(f"Replay {dataset.id}")
    for i in range(episodes):
        trajectory = dataset.episodes[i]
        env.reset(seed=seed + i)
        for action in trajectory.actions:
            env.step(action)
        if (i + 1) % print_interval == 0:
            print(f"probe episode={i + 1}/{episodes}")


def probe(
    agent: Agent,
    dataset: Dataset,
    probe: ProbeInterface,
    eval_fn: ProbeEvalFn,
    *,
    task_id: str | None = None,
    episodes: int = 1,
    seed: int = SEED,
    env_kwargs: dict | None = None,
) -> tuple[object, object]:
    task_id = task_id or _task_id(dataset.id, agent.id)
    env = dataset.make_env(**(env_kwargs or {}))
    probe_col = ProbeCollectWrapper(env, probe, eval_fn)
    print_stage(f"Probe {agent.id} in {dataset.id}")
    eval_col = EvalCollector(probe_col)
    replay(dataset, eval_col, episodes=episodes, seed=seed, print_interval=1)
    path = data_path("probe", task_id)
    eval_col.save(path)
    eval_col.close()
    probe_data = probe_col.save(path)
    return path, probe_data
