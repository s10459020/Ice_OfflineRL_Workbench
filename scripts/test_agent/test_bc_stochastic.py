import numpy as np
import torch
import gymnasium as gym

from ice_offline.agent.bc_stochastic import BCStochasticAgent
from ice_offline.dataset._lookup import make_dataset
from ice_offline.dataset.base import Dataset
from ice_offline.config.paths import data_path_test
from ice_offline.tools.printer import print_stage
from ice_offline.store.minari.collector import MinariCollectorWrapper
from ice_offline.store.state.hopper import HopperState, HopperStateIO
from ice_offline.store.state.op_collector import StateCollectWrapper


MODEL_STEP = 200_000
EPISODES = 10
SEED = 42
PRINT_INTERVAL = 1
AGENT_ID = "bc_stochastic"


def test(
    dataset: Dataset,
    *,
    episodes: int = EPISODES,
    model_step: int = MODEL_STEP,
    eval_env: gym.Env | None = None,
    seed: int = SEED,
    print_interval: int = PRINT_INTERVAL,
) -> list[float]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    eval_env = eval_env or dataset.make_env()

    print_stage("Test BC Stochastic")
    agent = BCStochasticAgent(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
    )
    agent.load(dataset.id, model_step)

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
    episodes: int = EPISODES,
    model_step: int = MODEL_STEP,
    seed: int = SEED,
    print_interval: int = PRINT_INTERVAL,
):
    env = dataset.make_env()
    state_col = StateCollectWrapper(env, state_cls=HopperState, state_io_cls=HopperStateIO)
    minari_col = MinariCollectorWrapper(state_col)

    returns = test(
        dataset=dataset,
        episodes=episodes,
        model_step=model_step,
        eval_env=minari_col,
        seed=seed,
        print_interval=print_interval,
    )

    data_path = data_path_test(dataset.id, AGENT_ID)
    minari_data = minari_col.save(data_path, id=dataset.id, agent_id=AGENT_ID)
    state_data = state_col.save(data_path)
    minari_col.close()

    return returns, minari_data, state_data


if __name__ == "__main__":
    dataset = make_dataset("hopper_simple", device="cuda")
    returns, minari_data, state_data = collect(
        dataset=dataset,
        episodes=EPISODES,
        seed=SEED,
        print_interval=PRINT_INTERVAL,
    )
    print(f"total_returns={sum(returns)}")
    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")






