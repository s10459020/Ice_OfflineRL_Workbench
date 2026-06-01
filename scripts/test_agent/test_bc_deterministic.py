
import numpy as np
import torch

import gymnasium as gym

from ice_offline.dataset._spec import Dataset
from ice_offline.dataset.hopper_simple import HopperSimpleDataset
from ice_offline.tools.printer import print_stage
from ice_offline.data.state.hopper import HopperState, HopperStateIO
from ice_offline.data.state.op_collector import StateCollectWrapper
from ice_offline.data.minari.collector import MinariCollectorWrapper
from ice_offline.agent.bc_deterministic import BCAgentDeterministic
from ice_offline.agent._spec import model_ref


MODEL_STEP = 200_000
EPISODES = 10
SEED = 42
PRINT_INTERVAL = 1

def train(
    dataset: Dataset,
    *,
    task_id: str = None,
    episodes: int = EPISODES,
    eval_env: gym.Env | None = None,
    seed: int = SEED,
    print_interval: int = 0,
) -> list[float]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    task_id = task_id or f"{dataset.env_id}_bc_deterministic-v0"
    eval_env = eval_env or dataset.make_env()

    print_stage("Test BC Deterministic")
    agent = BCAgentDeterministic(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
    )
    agent.load(model_ref(task_id, MODEL_STEP))

    returns = []
    for episode in range(1, episodes + 1):
        rewards = []
        o, _ = eval_env.reset(seed=seed + episode)
        while True:
            a = agent.act(o)
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
    dataset: Dataset,
    *,
    task_id: str = None,
    episodes: int = EPISODES,
    seed: int = SEED,
    print_interval: int = 0,
) -> None:
    env = dataset.make_env()
    state_col = StateCollectWrapper(env, state_cls=HopperState, state_io_cls=HopperStateIO)
    minari_col = MinariCollectorWrapper(state_col)

    returns = train(
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
    dataset = HopperSimpleDataset().load()
    returns, minari_data, state_data = collect(
        dataset=dataset, 
        task_id=f"{dataset.id}-bc_deterministic-v0", 
        episodes=EPISODES, 
        seed=SEED,
        print_interval=PRINT_INTERVAL,
    )
    print(f"avg_returns={sum(returns) / len(returns):.2f}")
    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")
