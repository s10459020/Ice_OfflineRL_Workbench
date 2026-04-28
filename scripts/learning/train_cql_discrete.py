from __future__ import annotations

import minari
import numpy as np

from ice_offline.agent import CQLAgentDiscrete
from ice_offline.tools.printer import print_stage

DATASET_ID = "minigrid/BabyAI-OneRoomS8/optimal-fullobs-v0"
BATCH_SIZE = 64
TRAIN_STEPS = 50_000
LOG_EVERY_STEPS = 1_000


def _build_transitions(dataset: minari.MinariDataset) -> dict[str, np.ndarray]:
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    rew_list: list[np.ndarray] = []
    next_obs_list: list[np.ndarray] = []
    done_list: list[np.ndarray] = []

    for episode in dataset.iterate_episodes():
        obs = np.asarray(episode.observations, dtype=np.float32)
        act = np.asarray(episode.actions)
        rew = np.asarray(episode.rewards, dtype=np.float32)
        term = np.asarray(episode.terminations, dtype=np.float32)
        trunc = np.asarray(episode.truncations, dtype=np.float32)

        done = np.clip(term + trunc, 0.0, 1.0)
        obs_list.append(obs[:-1])
        next_obs_list.append(obs[1:])
        act_list.append(act)
        rew_list.append(rew)
        done_list.append(done)

    return {
        "obs": np.concatenate(obs_list, axis=0).astype(np.float32),
        "act": np.concatenate(act_list, axis=0).astype(np.int64),
        "rew": np.concatenate(rew_list, axis=0).astype(np.float32),
        "next_obs": np.concatenate(next_obs_list, axis=0).astype(np.float32),
        "done": np.concatenate(done_list, axis=0).astype(np.float32),
    }


def _sample_batch(data: dict[str, np.ndarray], batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    idx = rng.integers(0, data["obs"].shape[0], size=(batch_size,))
    return {
        "obs": data["obs"][idx],
        "act": data["act"][idx],
        "rew": data["rew"][idx],
        "next_obs": data["next_obs"][idx],
        "done": data["done"][idx],
    }


def main() -> None:
    print_stage("Load")
    dataset = minari.load_dataset(DATASET_ID)
    data = _build_transitions(dataset)

    obs_size = int(np.prod(data["obs"].shape[1:]))
    act_size = int(data["act"].max()) + 1
    data["obs"] = data["obs"].reshape(data["obs"].shape[0], -1)
    data["next_obs"] = data["next_obs"].reshape(data["next_obs"].shape[0], -1)

    print(f"dataset_id={DATASET_ID}")
    print(f"transitions={data['obs'].shape[0]}")
    print(f"obs_size={obs_size} act_size={act_size}")

    print_stage("Train")
    algo = CQLAgentDiscrete(obs_size=obs_size, act_size=act_size)
    rng = np.random.default_rng(42)

    for step in range(1, TRAIN_STEPS + 1):
        batch = _sample_batch(data, BATCH_SIZE, rng)
        algo.update(batch, grad_step=step)

        if step % LOG_EVERY_STEPS == 0:
            print(f"step={step}/{TRAIN_STEPS}")

    print_stage("Done")
    print("train_complete=True")


if __name__ == "__main__":
    main()
