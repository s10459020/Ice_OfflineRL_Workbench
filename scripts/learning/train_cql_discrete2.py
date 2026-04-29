from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import FullyObsWrapper

from ice_offline.agent import CQLAgentDiscrete
from ice_offline.runner import OfflineRunner
from ice_offline.tools.printer import print_stage

DATASET_ID = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"
ENV_ID = "BabyAI-OneRoomS8-v0"
BATCH_SIZE = 64
TRAIN_STEPS = 50_000
LOG_EVERY_STEPS = 1_000
EVAL_BATCHES = 8
EVAL_EPISODES = 5


def obs_encode(obs: dict[str, np.ndarray]) -> np.ndarray:
    obs_arr = np.asarray(obs["image"], dtype=np.float32)
    return obs_arr.reshape(obs_arr.shape[0], -1)

def eval_env() -> gym.Env:
    return FullyObsWrapper(gym.make(ENV_ID))

def eval_loss(agent: CQLAgentDiscrete, episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    o, a, r, on, d = episode_batch
    return {"loss": float(agent.loss_critic(o, a, r, on, d).item())}

def eval_loss_td(agent: CQLAgentDiscrete, episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    o, a, r, on, d = episode_batch
    return {"loss_td": float(agent._loss_td(o, a, r, on, d).item())}

def eval_loss_cql(agent: CQLAgentDiscrete, episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    o, a, _, _, _ = episode_batch
    return {"loss_cql": float(agent._loss_cql(o, a).item())}

def eval_reward(episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    _, _, r, _, _ = episode_batch
    return {"return": float(r.sum().item())}


def main() -> None:
    print_stage("Load")
    runner = OfflineRunner(
        dataset_id=DATASET_ID,
        obs_encode=obs_encode,
        batch_size=BATCH_SIZE,
        train_steps=TRAIN_STEPS,
        log_every_steps=LOG_EVERY_STEPS,
        eval_batches=EVAL_BATCHES,
        eval_episodes=EVAL_EPISODES,
    )
    dataset = runner.load_dataset()
    agent = CQLAgentDiscrete(obs_size=dataset.obs_size, act_size=dataset.act_size)

    print(f"dataset_id={DATASET_ID}")
    print(f"transitions={dataset.num_transitions}")
    print(f"obs_size={dataset.obs_size} act_size={dataset.act_size}")

    print_stage("Train")
    runner.train(
        agent=agent,
        dataset=dataset,
        eval_offline_fns=[eval_loss, eval_loss_td, eval_loss_cql],
        eval_online_fns=[eval_reward],
        eval_env_fn=eval_env,
    )

    print_stage("Done")
    print("train_complete=True")


if __name__ == "__main__":
    main()

