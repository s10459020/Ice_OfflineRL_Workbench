from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import torch


@dataclass
class TorchBuffer:
    obs_list: torch.Tensor
    next_obs_list: torch.Tensor
    act_list: torch.Tensor
    rew_list: torch.Tensor
    done_list: torch.Tensor


@dataclass
class Dataset:
    # ====================
    # Dataset identity
    # ====================
    id: str
    env_id: str = ""
    dataset_path: str | Path = Path()
    seed: int = 0
    device: str = "cpu"

    # ====================
    # Loaded dataset info
    # ====================
    buffer: TorchBuffer = field(init=False)
    obs_shape: tuple[int, ...] = ()
    act_shape: tuple[int, ...] = ()
    obs_dim: int = 0
    act_dim: int = 0
    count: int = 0

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        torch.manual_seed(self.seed)

    # ====================
    # Public API
    # ====================
    def load(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)

        loader = self.make_loader()
        self.buffer = loader.buffer
        self.obs_shape = loader.obs_shape
        self.act_shape = loader.act_shape
        self.obs_dim = loader.obs_dim
        self.act_dim = loader.act_dim
        self.count = loader.count
        if not self.env_id:
            self.env_id = loader.env_id

        return self

    def set_seed(self, seed: int | None = None) -> None:
        torch.manual_seed(seed)

    def sample_batch(self, batch_size: int):
        idx = torch.randint(self.count, (batch_size,), device=self.buffer.obs_list.device)
        return TorchBuffer(
            obs_list=self.buffer.obs_list[idx],
            next_obs_list=self.buffer.next_obs_list[idx],
            act_list=self.buffer.act_list[idx],
            rew_list=self.buffer.rew_list[idx],
            done_list=self.buffer.done_list[idx],
        )

    # ====================
    # Loader binding
    # ====================
    def make_loader(self):
        from ice_offline.data.minari.loader import MinariLoader
        loader = MinariLoader(self.dataset_path, device=self.device)
        return loader

    # ====================
    # Env factory
    # ====================
    def make_env(self):
        return gym.make(self.env_id)

    def make_eval_env(self):
        return gym.make(self.env_id)

    def make_render_env(self):
        return gym.make(self.env_id, render_mode="human")
