import numpy as np
import torch
import gymnasium as gym

from ice_offline.dataset._lookup import get_dataset
from ice_offline.dataset._spec import BaseDataset
from ice_offline.tools.printer import print_stage
from ice_offline.pipeline.minari.collector import MinariCollectorWrapper
from ice_offline.pipeline.state.hopper import HopperState, HopperStateIO
from ice_offline.pipeline.state.op_collector import StateCollectWrapper


DATASET_KEY = "hopper_simple"
PRINT_INTERVAL = 1
EPISODES = 10
SEED = 42


def test(
    dataset: BaseDataset,
    *,
    task_id: str = None,
    episodes: int = EPISODES,
    eval_env: gym.Env | None = None,
    seed: int = SEED,
    print_interval: int = 0,
) -> list[float]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    task_id = task_id or f"{dataset.env_id}_random-v0"
    eval_env = eval_env or dataset.make_collect_env()

    print_stage("Test RANDOM")
    returns = []
    for episode in range(1, episodes + 1):
        rewards = []
        o, _ = eval_env.reset(seed=seed + episode)
        while True:
            a = eval_env.action_space.sample()
            o, r, terminated, truncated, _ = eval_env.step(a)
            rewards.append(r)
            if terminated or truncated:
                break
        total_reward = sum(rewards)
        returns.append(total_reward)

        if print_interval and episode % print_interval == 0:
            print(f"Episode {episode}/{episodes}, Return: {total_reward:.2f}")

    return returns


def collect(
    dataset: BaseDataset,
    *,
    task_id: str = None,
    episodes: int = EPISODES,
    seed: int = SEED,
    print_interval: int = 0,
):
    task_id = task_id or f"{dataset.env_id}_random-v0"
    env = dataset.make_collect_env()
    state_col = StateCollectWrapper(env, state_cls=HopperState, state_io_cls=HopperStateIO)
    minari_col = MinariCollectorWrapper(state_col)

    returns = test(
        dataset=dataset,
        task_id=task_id,
        episodes=episodes,
        eval_env=minari_col,
        seed=seed,
        print_interval=print_interval,
    )

    minari_data = minari_col.save(f"test/{task_id}")
    state_data = state_col.save(f"test/{task_id}")
    minari_col.close()

    return returns, minari_data, state_data


if __name__ == "__main__":
    dataset = get_dataset(DATASET_KEY)
    returns, minari_data, state_data = collect(
        dataset=dataset,
        task_id=f"{DATASET_KEY}_random-v0",
        episodes=EPISODES,
        seed=SEED,
        print_interval=PRINT_INTERVAL,
    )
    print(f"total_returns={sum(returns)}")
    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")
