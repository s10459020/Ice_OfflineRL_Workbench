from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import FullyObsWrapper

from ice_offline.agent import CQLAgentDiscrete
from ice_offline.dataset import load_minari
from ice_offline.tools.printer import print_stage

DATASET_ID = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"
BATCH_SIZE = 64
TRAIN_STEPS = 50_000
LOG_EVERY_STEPS = 1_000
EVAL_BATCHES = 8
EVAL_EPISODES = 5
ENV_ID = "BabyAI-OneRoomS8-v0"


def encode_obs(obs: np.ndarray | dict[str, np.ndarray]) -> np.ndarray:
    obs_arr = np.asarray(obs["image"], dtype=np.float32)
    return obs_arr.reshape(obs_arr.shape[0], -1)


def evaluate_loss(algo: CQLAgentDiscrete, dataset, batch_size: int, n_batches: int) -> float:
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(n_batches):
            batch = dataset.sample_batch(batch_size)
            o = torch.as_tensor(batch["obs"], dtype=torch.float32, device=algo.device)
            a = torch.as_tensor(batch["act"], dtype=torch.long, device=algo.device).view(-1)
            r = torch.as_tensor(batch["rew"], dtype=torch.float32, device=algo.device).view(-1, 1)
            on = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=algo.device)
            d = torch.as_tensor(batch["done"], dtype=torch.float32, device=algo.device).view(-1, 1)
            losses.append(float(algo._loss(o, a, r, on, d).item()))
    return float(np.mean(losses))


def evaluate_reward(algo: CQLAgentDiscrete, env_id: str, n_episodes: int) -> float:
    env = FullyObsWrapper(gym.make(env_id))
    returns: list[float] = []
    try:
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            while not done:
                obs_vec = encode_obs({"image": np.asarray([obs["image"]])})[0]
                action = algo.act(obs_vec, epsilon=0.0)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)
                done = bool(terminated or truncated)
            returns.append(ep_return)
    finally:
        env.close()
    return float(np.mean(returns))


def main() -> None:
    print_stage("Load")
    dataset = load_minari(DATASET_ID, obs_transform=encode_obs)
    obs_size = dataset.obs_size
    act_size = dataset.act_size

    print(f"dataset_id={DATASET_ID}")
    print(f"transitions={dataset.num_transitions}")
    print(f"obs_size={obs_size} act_size={act_size}")

    print_stage("Train")
    algo = CQLAgentDiscrete(obs_size=obs_size, act_size=act_size)

    for step in range(1, TRAIN_STEPS + 1):
        batch = dataset.sample_batch(BATCH_SIZE)
        algo.update(batch, grad_step=step)

        if step % LOG_EVERY_STEPS == 0:
            eval_loss = evaluate_loss(algo, dataset, BATCH_SIZE, EVAL_BATCHES)
            eval_return = evaluate_reward(algo, ENV_ID, EVAL_EPISODES)
            print(
                f"step={step}/{TRAIN_STEPS} "
                f"eval_loss={eval_loss:.6f} eval_return={eval_return:.4f}"
            )

    print_stage("Done")
    print("train_complete=True")


if __name__ == "__main__":
    main()
