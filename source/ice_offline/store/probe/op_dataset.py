from pathlib import Path

import h5py
import numpy as np


class ProbeDataset:
    # ====================
    # Init
    # ====================
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self.path, "a")
        self._indices = self._read_indices(self._h5)
        self.episode_count = len(self._indices)
        self.step_counts = self._read_counts(self._h5, self._indices)

    # ====================
    # Public API
    # ====================
    @classmethod
    def load_dataset(cls, path: Path) -> "ProbeDataset":
        return cls(path=path)

    def close(self) -> None:
        self._h5.close()

    def read_episode(self, episode_index: int) -> list[dict[str, np.ndarray]]:
        episode = self._h5[self._indices[episode_index]]
        keys = list(episode.keys())
        values = {key: episode[key][()] for key in keys}
        length = int(values[keys[0]].shape[0])
        return [{key: values[key][step_index] for key in keys} for step_index in range(length)]

    def read_step(self, episode_index: int, step_index: int) -> dict[str, np.ndarray]:
        episode = self._h5[self._indices[episode_index]]
        return {key: episode[key][step_index] for key in episode.keys()}

    @classmethod
    def write(cls, path: Path, episodes: list[list[dict[str, np.ndarray]]]) -> "ProbeDataset":
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h5_file:
            for episode_index, sequence in enumerate(episodes):
                group_name = f"episode_{episode_index}"
                episode_group = h5_file.create_group(group_name)
                keys = sequence[0].keys()
                for key in keys:
                    values = [item[key] for item in sequence]
                    episode_group.create_dataset(key, data=np.asarray(values))
        return cls(path=path)

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
            probe_length = int(episode_group[first_key].shape[0])
            counts.append(max(0, probe_length - 1))
        return counts
