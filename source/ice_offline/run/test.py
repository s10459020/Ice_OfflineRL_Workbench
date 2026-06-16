import gymnasium as gym
import numpy as np
import torch

from ice_offline.agent._spec import Agent
from ice_offline.config.paths import _task_id
from ice_offline.dataset.base import Dataset
from ice_offline.tools.printer import print_stage


EPISODES = 10
PRINT_INTERVAL = 1
SEED = 42


def test(
    agent: Agent,
    dataset: Dataset,
    *,
    task_id: str | None = None,
    episodes: int = EPISODES,
    seed: int = SEED,
    print_interval: int = PRINT_INTERVAL,
    env_kwargs: dict | None = None,
) -> list[float]:
    _ = task_id or _task_id(dataset.id, agent.id)
    env = dataset.make_env(**(env_kwargs or {}))

    print_stage(f"Test {agent.id} in {dataset.id}")
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

    env.close()
    return returns
