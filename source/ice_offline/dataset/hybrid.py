import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ice_offline.dataset._types import Buffer
from ice_offline.dataset._types import Episode
from ice_offline.dataset._types import Metadata
from ice_offline.dataset.base import Dataset
from ice_offline.store.minari.loader import MinariLoader


@dataclass(kw_only=True)
class HybridDataset(Dataset):
    sample_count: int
    dataset_a: Dataset
    dataset_b: Dataset
    random_ratio: float = 0.5

    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)
        self.loader = MinariLoader(path=self.path, device=self.device)
        self.env_id = self.dataset_a.env_id
        self._episodes = self.hybridEpisodes()
        self._buffer = self.loader.buffer_from_episodes(self._episodes, device=self.device)
        self._metadata = Metadata(
            env_id=self.env_id,
            obs_shape=self.dataset_a.obs_shape,
            act_shape=self.dataset_a.act_shape,
            obs_dim=self.dataset_a.obs_dim,
            act_dim=self.dataset_a.act_dim,
            count=self.sample_count,
        )

    def hybridEpisodes(self) -> list[Episode]:
        rng = np.random.default_rng(self.seed)
        count_a = int(self.sample_count * self.random_ratio)
        count_b = self.sample_count - count_a
        episodes_a = self._sampleEpisodes(self.dataset_a.episodes, count_a, rng)
        episodes_b = self._sampleEpisodes(self.dataset_b.episodes, count_b, rng)
        return episodes_a + episodes_b

    def hybridBuffer(self) -> Buffer:
        return self.loader.buffer_from_episodes(self.episodes, device=self.device)

    def _sampleEpisodes(
        self,
        source: list[Episode],
        count: int,
        rng: np.random.Generator,
    ) -> list[Episode]:
        total = 0
        episodes: list[Episode] = []
        while total < count:
            episode = source[int(rng.integers(len(source)))]
            episode_steps = int(len(episode.rewards))
            remain = count - total
            if episode_steps <= remain:
                episodes.append(self._copyEpisode(episode))
                total += episode_steps
            else:
                episodes.append(self._truncateEpisode(episode, remain))
                total += remain
        return episodes

    def _truncateEpisode(self, episode: Episode, count: int) -> Episode:
        terminations = np.asarray(episode.terminations[:count], dtype=np.bool_).copy()
        truncations = np.asarray(episode.truncations[:count], dtype=np.bool_).copy()
        terminations[-1] = False
        truncations[-1] = True
        return Episode(
            observations=self._sliceNode(episode.observations, count + 1),
            actions=np.asarray(episode.actions[:count]).copy(),
            rewards=np.asarray(episode.rewards[:count]).copy(),
            terminations=terminations,
            truncations=truncations,
            infos=self._sliceInfos(episode.infos, count),
        )

    def _copyEpisode(self, episode: Episode) -> Episode:
        count = int(len(episode.rewards))
        return Episode(
            observations=self._sliceNode(episode.observations, count + 1),
            actions=np.asarray(episode.actions).copy(),
            rewards=np.asarray(episode.rewards).copy(),
            terminations=np.asarray(episode.terminations, dtype=np.bool_).copy(),
            truncations=np.asarray(episode.truncations, dtype=np.bool_).copy(),
            infos=self._sliceInfos(episode.infos, count),
        )

    def _sliceNode(self, value, count: int):
        if isinstance(value, dict):
            return {key: self._sliceNode(item, count) for key, item in value.items()}
        return np.asarray(value[:count]).copy()

    def _sliceInfos(self, infos, count: int):
        if infos is None:
            return None
        return self._sliceNode(infos, count)

    def save(self, path: Path, dataset_id: str) -> None:
        self.loader.write_episodes(path, self.episodes)

        metadata_path = path.with_name("metadata.json")
        metadata = asdict(self._metadata)
        metadata.update(
            dataset_id=dataset_id,
            source_dataset_id_a=self.dataset_a.id,
            source_dataset_id_b=self.dataset_b.id,
            random_ratio=self.random_ratio,
            seed=self.seed,
        )
        with metadata_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
            file.write("\n")
