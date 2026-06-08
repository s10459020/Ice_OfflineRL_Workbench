import gymnasium as gym
import numpy as np
import torch

from ice_offline.agent.scas_aspl import ScasAsplAgent
from ice_offline.agent.scas_min import ScasDynamic
from ice_offline.config.paths import data_path_collect
from ice_offline.dataset._spec import Dataset
from ice_offline.dataset.hopper_expert import HopperExpertDataset
from ice_offline.dataset.loader.minari.collector import MinariCollectorWrapper
from ice_offline.tools.printer import print_stage


DATASET_CLASS = HopperExpertDataset
MODEL_STEP = 200_000
DYNAMIC_STEP = 100_000
EPISODES = 1
SEED = 42
PRINT_INTERVAL = 1
AGENT_ID = "scas_aspl"
DEVICE = "cuda:0"


def test(
    dataset: Dataset,
    *,
    episodes: int = EPISODES,
    model_step: int = MODEL_STEP,
    dynamic_step: int = DYNAMIC_STEP,
    eval_env: gym.Env | None = None,
    seed: int = SEED,
    print_interval: int = PRINT_INTERVAL,
) -> list[float]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    eval_env = eval_env or dataset.make_env()

    print_stage("Collect SCAS ASPL")
    dynamics = ScasDynamic(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        device=dataset.device,
    )
    dynamics.agent_name = f"{AGENT_ID}_dynamics"
    dynamics.load(dataset.id, dynamic_step)

    agent = ScasAsplAgent(
        obs_size=dataset.obs_dim,
        act_size=dataset.act_dim,
        dynamics=dynamics,
        device=dataset.device,
    )
    agent.load(dataset.id, model_step)

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
    episodes: int = EPISODES,
    model_step: int = MODEL_STEP,
    dynamic_step: int = DYNAMIC_STEP,
    seed: int = SEED,
    print_interval: int = PRINT_INTERVAL,
):
    env = dataset.make_env()
    minari_col = MinariCollectorWrapper(env)

    returns = test(
        dataset=dataset,
        episodes=episodes,
        model_step=model_step,
        dynamic_step=dynamic_step,
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
        episodes=EPISODES,
        seed=SEED,
        print_interval=PRINT_INTERVAL,
    )
    print(f"avg_returns={sum(returns) / len(returns):.2f}")
    print(f"dataset_id={minari_data.spec.dataset_id}")
    print(f"total_episodes={minari_data.total_episodes}")
    print(f"total_steps={minari_data.total_steps}")
