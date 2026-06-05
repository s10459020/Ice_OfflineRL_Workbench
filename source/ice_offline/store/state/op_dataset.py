from pathlib import Path
from typing import Type

import h5py
import numpy as np

from ice_offline.store.state._spec import State


class StateDataset:
    # ====================
    # Init
    # ====================
    def __init__(self, path: Path, state_cls: Type[State]) -> None:
        self.path = path
        self.state_cls = state_cls
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self.path, "a")
        self._indices = self._read_indices(self._h5)
        self.episode_count = len(self._indices)
        self.step_counts = self._read_counts(self._h5, self._indices)

    # ====================
    # Public API
    # ====================
    @classmethod
    def load_dataset(cls, path: Path, state_cls: Type[State]) -> "StateDataset":
        return cls(path=path, state_cls=state_cls)

    def close(self) -> None:
        self._h5.close()

    def read_episode(self, episode_index: int) -> list[State]:
        episode = self._h5[self._indices[episode_index]]
        keys = list(episode.keys())
        values = {key: episode[key][()] for key in keys}
        length = int(values[keys[0]].shape[0])
        return [self.state_cls.from_serialized({key: values[key][i] for key in keys}) for i in range(length)]

    def read_step(self, episode_index: int, step_index: int) -> State:
        episode = self._h5[self._indices[episode_index]]
        payload = {key: episode[key][step_index] for key in episode.keys()}
        return self.state_cls.from_serialized(payload)

    def append_episode(self, sequence: list[dict[str, np.ndarray]]) -> None:
        next_index = self.episode_count
        group_name = f"episode_{next_index}"
        episode_group = self._h5.create_group(group_name)
        keys = sequence[0].keys()
        for key in keys:
            values = [item[key] for item in sequence]
            episode_group.create_dataset(key, data=np.asarray(values))
        self._indices.append(group_name)
        self.episode_count += 1
        self.step_counts.append(max(0, len(sequence) - 1))

    def append_episodes(self, episodes: list[list[dict[str, np.ndarray]]]) -> None:
        for sequence in episodes:
            self.append_episode(sequence)

    @classmethod
    def write(
        cls,
        path: Path,
        state_cls: Type[State],
        episodes: list[list[dict[str, np.ndarray]]],
    ) -> "StateDataset":
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h5_file:
            for episode_index, sequence in enumerate(episodes):
                group_name = f"episode_{episode_index}"
                episode_group = h5_file.create_group(group_name)
                keys = sequence[0].keys()
                for key in keys:
                    values = [item[key] for item in sequence]
                    episode_group.create_dataset(key, data=np.asarray(values))
        return cls(path=path, state_cls=state_cls)

    # ====================
    # Private
    # ====================
    def _read_indices(self, h5_file: h5py.File) -> list[str]:
        keys = [key for key in h5_file.keys() if key.startswith("episode_")]
        return sorted(keys, key=lambda key: int(key.split("_")[1]))

    def _read_counts(self, h5_file: h5py.File, indices: list[str]) -> list[int]:
        counts: list[int] = []
        for key in indices:
            episode_group = h5_file[key]
            first_key = next(iter(episode_group.keys()))
            state_length = int(episode_group[first_key].shape[0])
            counts.append(max(0, state_length - 1))
        return counts

