import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import torch

from ice_offline.dataset._types import Buffer, Metadata
from ice_offline.dataset.base import Dataset
from ice_offline.store.d4rl.loader import D4rlLoader


@dataclass(kw_only=True)
class HybridDataset(Dataset):
    sample_count: int
    dataset_a: Dataset
    dataset_b: Dataset
    random_ratio: float = 0.5

    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)
        self.env_id = self.dataset_a.env_id
        self._episodes = []
        self._buffer = self.hybridBuffer()
        self._metadata = Metadata(
            env_id=self.env_id,
            obs_shape=self.dataset_a.obs_shape,
            act_shape=self.dataset_a.act_shape,
            obs_dim=self.dataset_a.obs_dim,
            act_dim=self.dataset_a.act_dim,
            count=self.sample_count,
        )

    def hybridBuffer(self) -> Buffer:
        count_a = int(self.sample_count * self.random_ratio)
        count_b = self.sample_count - count_a

        source_a = self._sampleBuffer(self.dataset_a.buffer, count_a)
        source_b = self._sampleBuffer(self.dataset_b.buffer, count_b)
        return self._concatBuffer(source_a, source_b)

    def _sampleBuffer(self, source: Buffer, count: int) -> Buffer:
        indices = torch.randint(source.actions.shape[0], (count,), device=source.actions.device)
        return Buffer(
            observations=source.observations[indices].to(self.device),
            next_observations=source.next_observations[indices].to(self.device),
            actions=source.actions[indices].to(self.device),
            rewards=source.rewards[indices].to(self.device),
            dones=source.dones[indices].to(self.device),
        )

    def _concatBuffer(self, buffer_a: Buffer, buffer_b: Buffer) -> Buffer:
        return Buffer(
            observations=torch.cat([buffer_a.observations, buffer_b.observations], dim=0),
            next_observations=torch.cat([buffer_a.next_observations, buffer_b.next_observations], dim=0),
            actions=torch.cat([buffer_a.actions, buffer_b.actions], dim=0),
            rewards=torch.cat([buffer_a.rewards, buffer_b.rewards], dim=0),
            dones=torch.cat([buffer_a.dones, buffer_b.dones], dim=0),
        )

    def save(self, path: Path, dataset_id: str) -> None:
        D4rlLoader(path, device=self.device).write_buffer(path, self.buffer)

        metadata_path = path.with_name("metadata.json")
        metadata = asdict(self._metadata)
        metadata.update(
            dataset_id=dataset_id,
            source_dataset_id_a=self.dataset_a.id,
            source_dataset_id_b=self.dataset_b.id,
            random_ratio=self.random_ratio,
        )
        with metadata_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
            file.write("\n")
