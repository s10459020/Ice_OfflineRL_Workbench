from dataclasses import dataclass, field

import torch

from ice_offline.dataset._types import Episode, Metadata
from ice_offline.dataset.base import Dataset
from ice_offline.store.eval.loader import EvalLoader


@dataclass
class EvalDataset(Dataset):
    _batch_episodes: list[tuple[int, list[Episode]]] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)
        if self.loader is None:
            self.loader = self.make_loader()
        self._metadata = Metadata(env_id=self.env_id)

    @property
    def batch_episodes(self) -> list[tuple[int, list[Episode]]]:
        if self._batch_episodes is None:
            self._batch_episodes = self.loader.load_batch_episodes()
        return self._batch_episodes

    @property
    def episodes(self) -> list[Episode]:
        return [
            episode
            for _, episodes in self.batch_episodes
            for episode in episodes
        ]

    def make_loader(self):
        return EvalLoader(self.path, device=self.device)
