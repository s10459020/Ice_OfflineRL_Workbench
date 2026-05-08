import gymnasium as gym
import minari
import numpy as np
import torch
from collections import deque


def eval_return(episode_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float]:
    _, _, reward, _, _ = episode_batch
    return {"return": float(reward.sum().item())}


class StopReturnStable:
    def __init__(self, patience: int = 5, lambda_ratio: float = 0.01) -> None:
        self.patience = patience
        self.lambda_ratio = lambda_ratio
        self.recent = deque(maxlen=patience)

    def should_stop(self, metrics: dict[str, list[float]]) -> bool:
        self.recent.append(float(np.mean(metrics["return"])))
        if len(self.recent) < self.patience:
            return False

        seq = list(self.recent)
        base = abs(seq[0]) + 1e-12
        for i in range(1, len(seq)):
            ratio = abs(seq[i] - seq[i - 1]) / base
            if ratio > self.lambda_ratio:
                return False
        return True

class BaseDataset:
    dataset_name: str
    dataset_id: str
    env_id: str

    def obs_encode_batch(self, obs):
        return np.asarray(obs)

    def obs_encode(self, obs):
        return np.asarray(obs)

    def act_encode_batch(self, act):
        return np.asarray(act)

    def act_encode(self, act):
        return np.asarray(act)

    def make_dataset(self):
        return minari.load_dataset(self.dataset_id, download=True)

    def make_eval_env(self):
        return gym.make(self.env_id)

    def make_collect_env(self):
        return gym.make(self.env_id)

    def make_render_env(self):
        return gym.make(self.env_id, render_mode="human")

    def eval_online_fns(self):
        return [eval_return]

    def observation_cardinality(self, observation_shape: tuple[int, ...], minari_dataset) -> tuple[int, ...] | None:
        return None

    def action_cardinality(self, action_shape: tuple[int, ...], minari_dataset) -> tuple[int, ...] | None:
        if action_shape != (1,):
            return None
        n = getattr(minari_dataset.spec.action_space, "n", None)
        if n is None:
            return None
        return (int(n),)
