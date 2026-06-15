import gymnasium as gym
import numpy as np
import torch

from ice_offline.agent._spec import Agent
from ice_offline.config.paths import _task_id
from ice_offline.config.paths import data_path_test
from ice_offline.dataset.base import Dataset
from ice_offline.store.minari.collector import MinariCollectorWrapper
from ice_offline.store.state._lookup import STATE_OPS
from ice_offline.store.state.op_collector import StateCollectWrapper
from ice_offline.tools.printer import print_stage


EPISODES = 10
PRINT_INTERVAL = 1
SEED = 42


def rollout(
    agent: Agent,
    env: gym.Env,
    *,
    episodes: int = EPISODES,
    seed: int = SEED,
    print_interval: int = PRINT_INTERVAL,
) -> list[float]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    returns: list[float] = []
    for episode in range(1, episodes + 1):
        now_seed = seed + episode
        agent.set_seed(now_seed)
        rewards: list[float] = []
        observation, _ = env.reset(seed=now_seed)
        while True:
            action = agent.act(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break
        total_reward = sum(rewards)
        returns.append(total_reward)

        if print_interval > 0 and episode % print_interval == 0:
            print(f"test episode={episode}/{episodes} return={total_reward:.2f}")

    return returns


def test(
    agent: Agent,
    dataset: Dataset,
    *,
    task_id: str | None = None,
    episodes: int = EPISODES,
    seed: int = SEED,
    print_interval: int = PRINT_INTERVAL,
):
    task_id = task_id or _task_id(dataset.id, agent.id)
    if dataset.env_id not in STATE_OPS:
        raise ValueError(f"unsupported test state env: {dataset.env_id}")

    state_cls, state_io_cls, _ = STATE_OPS[dataset.env_id]
    env = dataset.make_env()
    state_col = StateCollectWrapper(env, state_cls=state_cls, state_io_cls=state_io_cls)
    minari_col = MinariCollectorWrapper(state_col)

    print_stage(f"Test {agent.id} in {dataset.id}")
    returns = rollout(
        agent=agent,
        env=minari_col,
        episodes=episodes,
        seed=seed,
        print_interval=print_interval,
    )

    path = data_path_test(task_id)
    minari_data = minari_col.save(path, id=dataset.id, agent_id=agent.id)
    state_data = state_col.save(path)
    minari_col.close()
    return returns, minari_data, state_data
