from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import torch

from ice_offline.dataset._types import Batch, Buffer, Episode, Metadata


@dataclass
class Dataset:
    # ====================
    # Dataset identity
    # ====================
    id: str = ""
    env_id: str = ""
    path: Path | None = None
    seed: int = 0
    device: str = "cuda"

    # ====================
    # Loaded dataset info
    # ====================
    loader: object | None = None
    _buffer: Buffer | None = field(init=False, default=None)
    _episodes: list[Episode] | None = field(init=False, default=None)
    _metadata: Metadata | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)
        if self.loader is None:
            self.loader = self.make_loader()
        self._metadata = self.loader.load_metadata()
        if not self.env_id:
            self.env_id = self._metadata.env_id

    # ====================
    # Public API
    # ====================
    @property
    def buffer(self) -> Buffer:
        if self._buffer is None:
            self._buffer = self.loader.load_buffer()
        return self._buffer

    @property
    def episodes(self) -> list[Episode]:
        if self._episodes is None:
            self._episodes = self.loader.load_episodes()
        return self._episodes

    @property
    def metadata(self) -> Metadata:
        if self._metadata is None:
            self._metadata = self.loader.load_metadata()
        return self._metadata

    @property
    def obs_shape(self) -> tuple[int, ...]:
        return self.metadata.obs_shape

    @property
    def act_shape(self) -> tuple[int, ...]:
        return self.metadata.act_shape

    @property
    def obs_dim(self) -> int:
        return self.metadata.obs_dim

    @property
    def act_dim(self) -> int:
        return self.metadata.act_dim

    @property
    def count(self) -> int:
        return self.metadata.count

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def step_counts(self) -> list[int]:
        return [int(episode.rewards.shape[0]) for episode in self.episodes]

    def set_seed(self, seed: int | None = None) -> None:
        torch.manual_seed(seed)

    def sample_batch(self, batch_size: int) -> Batch:
        buffer = self.buffer
        idx = torch.randint(self.count, (batch_size,), device=self.device)
        return (
            buffer.observations[idx],
            buffer.actions[idx],
            buffer.rewards[idx],
            buffer.next_observations[idx],
            buffer.dones[idx],
        )

    # ====================
    # Loader binding
    # ====================
    def make_loader(self):
        from ice_offline.store.minari.loader import MinariLoader
        loader = MinariLoader(self.path, device=self.device)
        return loader

    # ====================
    # Env factory
    # ====================
    def make_env(self, **kwargs):
        return gym.make(self.env_id, **kwargs)

    def make_eval_env(self, **kwargs):
        return gym.make(self.env_id, **kwargs)

    def make_render_env(self):
        return gym.make(self.env_id, render_mode="human")
