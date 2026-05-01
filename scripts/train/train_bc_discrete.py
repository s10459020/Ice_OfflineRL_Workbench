
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import FullyObsWrapper

from ice_offline.agent import DiscreteBCAgent
from ice_offline.dataset import BatchLoader
from ice_offline.runner import TorchBatchOfflineRunner
from ice_offline.tools.paths import eval_root
from ice_offline.tools.printer import print_stage
from ice_offline.tools.timing import Timer




DATASET_ID = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"
ENV_ID = "BabyAI-OneRoomS8-v0"
RUNNER_ID = "bc_discrete_onerooms8"
BATCH_SIZE = 64
TRAIN_STEPS = 100_000
EVAL_INTERVAL = 2_000
EVAL_BATCHES = 8
EVAL_EPISODES = 3
MODEL_LOAD_STEP = 0
MODEL_SAVE_INTERVAL = 50_000

def obs_encode(obs: dict[str, np.ndarray]) -> np.ndarray:
    obs_arr = np.asarray(obs["image"], dtype=np.float32)
    return obs_arr.reshape(obs_arr.shape[0], -1)


def eval_env() -> gym.Env:
    return FullyObsWrapper(gym.make(ENV_ID))


def eval_loss(agent: DiscreteBCAgent, episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    o, a, _, _, _ = episode_batch
    return {"loss": float(agent.loss_actor(o, a).item())}


def eval_reward(episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    _, _, r, _, _ = episode_batch
    return {"return": float(r.sum().item())}


def main() -> None:
    print_stage("Load")
    dataset = BatchLoader.from_minari(dataset_id=DATASET_ID, obs_encode=obs_encode)
    obs_dim = int(np.prod(dataset.obs_shape))
    act_size = dataset.act_size
    runner = TorchBatchOfflineRunner(
        obs_encode=obs_encode,
        batch_size=BATCH_SIZE,
        train_steps=TRAIN_STEPS,
        eval_batches=EVAL_BATCHES,
        eval_episodes=EVAL_EPISODES,
        eval_interval=EVAL_INTERVAL,
        runner_id=RUNNER_ID,
        model_load_step=MODEL_LOAD_STEP,
        model_save_interval=MODEL_SAVE_INTERVAL,
    )
    agent = DiscreteBCAgent(obs_size=obs_dim, act_size=act_size)

    print_stage("Train")
    Timer.stopwatch(f"train::{RUNNER_ID}")
    runner.train(
        agent=agent,
        dataset=dataset,
        eval_offline_fns=[eval_loss],
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






