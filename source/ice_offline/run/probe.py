import gymnasium as gym
import numpy as np

from ice_offline.agent._spec import Agent
from ice_offline.config.paths import data_path
from ice_offline.dataset.base import Dataset
from ice_offline.store.minari.collector import MinariCollectorWrapper
from ice_offline.store.probe.op_collector import ProbeCollectWrapper
from ice_offline.store.probe.op_collector import ProbeEvalFn
from ice_offline.store.probe.op_collector import ProbeInterface
from ice_offline.store.state._lookup import STATE_OPS
from ice_offline.store.state.op_replayer import make_replayer
from ice_offline.tools.printer import print_stage


def eval_prop(agent: Agent, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return agent.eval(observations, actions, "Pi")


def replay(
    dataset: Dataset,
    env: gym.Env,
    *,
    episodes: int,
    seed: int | None = None,
    print_interval: int = 1,
) -> None:
    for i in range(episodes):
        _, info = env.reset(seed=None if seed is None else seed + i)
        episode_index = info["episode_index"]
        trajectory = dataset.episodes[episode_index]
        for action in trajectory.actions:
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        if (i + 1) % print_interval == 0:
            print(f"replay episode={i + 1}/{episodes}")


def probe(
    task_id: str,
    agent: Agent,
    dataset: Dataset,
    probe: ProbeInterface,
    eval_fn: ProbeEvalFn,
    *,
    episodes: int = 10,
    seed: int | None = None,
    env_kwargs: dict | None = None,
) -> object:
    state_cls, state_io_cls, _ = STATE_OPS[dataset.env_id]
    env = make_replayer(
        dataset=dataset,
        state_cls=state_cls,
        state_io_cls=state_io_cls,
        eval_env=dataset.make_env(**(env_kwargs or {})),
        render_mode=None,
    )
    probe_col = ProbeCollectWrapper(env, probe, agent, eval_fn)
    minari_col = MinariCollectorWrapper(probe_col)

    print_stage(f"Probe Replay {task_id}")
    replay(dataset, minari_col, episodes=episodes, seed=seed, print_interval=1)
    path = data_path("probe", task_id)
    minari_col.save(path, id=task_id, agent_id=agent.id)
    probe_data = probe_col.save(path)
    minari_col.close()
    return probe_data


if __name__ == "__main__":
    from ice_offline.agent._lookup import make_agent
    from ice_offline.dataset._lookup import make_dataset
    from ice_offline.store.probe.action_axis_probe import ActionAxisProbe

    device = "cuda:0"
    task_id = "check_run-v1"
    dataset = make_dataset("hopper_simple", device=device)
    agent = make_agent("bc_stochastic", dataset, device=device)
    agent.load(task_id, 20_000)

    probe_data = probe(
        task_id,
        agent,
        dataset,
        ActionAxisProbe(100),
        eval_prop,
    )
    print(probe_data.path)
