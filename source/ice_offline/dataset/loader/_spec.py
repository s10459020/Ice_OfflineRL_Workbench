from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from ice_offline.dataset._spec import TorchBuffer


class DatasetEpisode(Protocol):
    observations: Any
    actions: Any
    rewards: Any
    terminations: Any
    truncations: Any
    infos: Any


class DatasetLoader(Protocol):
    dataset_path: Path
    device: str
    env_id: str

    buffer: TorchBuffer
    obs_shape: tuple[int, ...]
    act_shape: tuple[int, ...]
    obs_dim: int
    act_dim: int
    count: int

    total_episodes: int
    episode_steps: list[int]
    total_steps: int

    def __getitem__(self, episode_index: int) -> DatasetEpisode:
        ...

    def iterate_episodes(self) -> Iterator[DatasetEpisode]:
        ...
