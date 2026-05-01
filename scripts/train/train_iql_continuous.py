
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ice_offline.agent import IQLAgentContinuous
from ice_offline.dataset import BatchLoader
from ice_offline.runner import TorchBatchOfflineRunner
from ice_offline.tools.paths import eval_root
from ice_offline.tools.printer import print_stage
from ice_offline.tools.timing import Timer




DATASET_ID = "mujoco/invertedpendulum/expert-v0"
ENV_ID = "InvertedPendulum-v5"
RUNNER_ID = "iql_continuous_invertedpendulum"
BATCH_SIZE = 64
TRAIN_STEPS = 1_000_000
EVAL_INTERVAL = 1_000
EVAL_BATCHES = 8
EVAL_EPISODES = 5
MODEL_LOAD_STEP = 0
MODEL_SAVE_INTERVAL = 50_000

def obs_encode(obs: np.ndarray) -> np.ndarray:
    obs_arr = np.asarray(obs, dtype=np.float32)
    return obs_arr.reshape(obs_arr.shape[0], -1)



def eval_loss_q(agent: IQLAgentContinuous, episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    o, a, r, on, d = episode_batch
    with torch.no_grad():
        return {"loss_q": float(agent._loss_q(o, a, r, on, d).item())}


def eval_loss_v(agent: IQLAgentContinuous, episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    o, a, _, _, _ = episode_batch
    with torch.no_grad():
        return {"loss_v": float(agent._loss_v(o, a).item())}


def eval_loss_pi(agent: IQLAgentContinuous, episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    o, a, _, _, _ = episode_batch
    with torch.no_grad():
        return {"loss_pi": float(agent.loss_actor(o, a).item())}


def eval_env() -> gym.Env:
    return gym.make(ENV_ID)


def eval_reward(episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    _, _, r, _, _ = episode_batch
    return {"reward_sum": float(r.sum().item())}


def main() -> None:
    print_stage("Load")
    dataset = BatchLoader.from_minari(dataset_id=DATASET_ID, obs_encode=obs_encode)
    obs_dim = int(np.prod(dataset.obs_shape))
    act_dim = int(np.prod(dataset.act_shape))
    runner = TorchBatchOfflineRunner(
        obs_encode=obs_encode,
        batch_size=BATCH_SIZE,
        train_steps=TRAIN_STEPS,
        eval_batches=EVAL_BATCHES,
        eval_episodes=5,
        eval_interval=EVAL_INTERVAL,
        runner_id=RUNNER_ID,
        model_load_step=MODEL_LOAD_STEP,
        model_save_interval=MODEL_SAVE_INTERVAL,
    )
    agent = IQLAgentContinuous(obs_size=obs_dim, act_size=act_dim)

    print_stage("Train")
    Timer.stopwatch(f"train::{RUNNER_ID}")
    runner.train(
        agent=agent,
        dataset=dataset,
        eval_offline_fns=[eval_loss_q, eval_loss_v, eval_loss_pi],
        eval_online_fns=[eval_reward],
        eval_env_fn=eval_env,
    )

    train_ms = Timer.stopwatch(f"train::{RUNNER_ID}")
    time_path = Path(eval_root()) / f"{RUNNER_ID}.txt"
    time_path.parent.mkdir(parents=True, exist_ok=True)
    time_path.write_text(
        f"runner_id={RUNNER_ID}\ntrain_ms={train_ms:.3f}\ntrain_sec={train_ms/1000.0:.3f}\n",
        encoding="utf-8",
    )

    print_stage("Done")


if __name__ == "__main__":
    main()






