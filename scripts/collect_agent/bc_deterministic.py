import gymnasium as gym
import numpy as np
import torch

from ice_offline.agent.bc_deterministic import BCDeterministicAgent
from ice_offline.config.paths import data_path_collect
from ice_offline.dataset._spec import Dataset
from ice_offline.dataset.hopper_expert import HopperExpertDataset
from ice_offline.dataset.hopper_medium import HopperMediumDataset
from ice_offline.dataset.hopper_simple import HopperSimpleDataset
from ice_offline.store.minari.collector import MinariCollectorWrapper
from ice_offline.tools.printer import print_stage


DATASET_CLASS = HopperExpertDataset
STEPS = 100_000
EPISODES = 1
SEED = 42
PRINT_INTERVAL = 1
AGENT_ID = "bc_deterministic"
DEVICE = "cuda:0"
BATCH_SIZE = 256


def train(
    dataset: Dataset,
    *,
    steps: int = STEPS,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset.set_seed(seed)

    print_stage("Train BC Deterministic")
    agent = BCDeterministicAgent(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        device=dataset.device,
    )

    for _ in range(steps):
        batch = dataset.sample_batch(batch_size)
        agent.update(batch)

    agent.save(dataset.id, steps)


def test(
    dataset: Dataset,
    *,
    episodes: int = EPISODES,
    steps: int = STEPS,
    eval_env: gym.Env | None = None,
    seed: int = SEED,
    print_interval: int = PRINT_INTERVAL,
) -> list[float]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    eval_env = eval_env or dataset.make_env()

    print_stage("Test BC Deterministic")
    agent = BCDeterministicAgent(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        device=dataset.device,
    )
    agent.load(dataset.id, steps)

    returns = []
    for episode in range(1, episodes + 1):
        rewards = []
        observation, _ = eval_env.reset(seed=seed)
        while True:
            action = agent.act(observation)
            observation, reward, terminated, truncated, _ = eval_env.step(action)
            rewards.append(reward)
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
    steps: int = STEPS,
    episodes: int = EPISODES,
    seed: int = SEED,
    print_interval: int = PRINT_INTERVAL,
):
    train(
        dataset=dataset,
        steps=steps,
        seed=seed,
    )

    env = dataset.make_env()
    minari_col = MinariCollectorWrapper(env)

    returns = test(
        dataset=dataset,
        episodes=episodes,
        steps=steps,
        eval_env=minari_col,
        seed=seed,
        print_interval=print_interval,
    )

    data_path = data_path_collect(dataset.id, AGENT_ID)
    minari_data = minari_col.save(data_path, id=dataset.id, agent_id=AGENT_ID)
    minari_col.close()

    return returns, minari_data


if __name__ == "__main__":
    dataset = DATASET_CLASS(device=DEVICE)
    returns, minari_data = collect(
        dataset=dataset,
        steps=STEPS,
        episodes=EPISODES,
        seed=SEED,
        print_interval=PRINT_INTERVAL,
    )
    print(f"avg_returns={sum(returns) / len(returns):.2f}")
    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")
