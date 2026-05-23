from pathlib import Path
from typing import Any, Type

import h5py
import minari
import numpy as np

from ice_offline.tools.paths import minari_root


class StateConverter:
    # ====================
    # Init
    # ====================
    def __init__(self, dataset_id: str, converter_cls: Type) -> None:
        self._dataset = minari.load_dataset(dataset_id, download=True)
        self._path = self._resolve_state_path(dataset_id)
        self._converter = converter_cls()

    # ====================
    # Public API
    # ====================
    def total_episodes(self) -> int:
        return self._dataset.total_episodes

    def reset(self) -> Path:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self._path, "w"):
            pass
        return self._path

    def convert(self, episode_index: int) -> Path:
        trajectory = self._dataset[episode_index]
        states = self._converter.convert_episode(trajectory)
        serialized_states = [state.serialize() for state in states]
        self._save_episode_data(episode_index, serialized_states)
        return self._path

    def convert_all(self) -> Path:
        self.reset()
        for episode_index in range(self._dataset.total_episodes):
            self.convert(episode_index)
        return self._path

    # ====================
    # Private
    # ====================
    def _save_episode_data(self, episode_index: int, sequence: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self._path, "a") as h5_file:
            group_name = f"episode_{episode_index}"
            if group_name in h5_file:
                del h5_file[group_name]
            episode_group = h5_file.require_group(group_name)
            keys = sequence[0].keys()
            for key in keys:
                values = [item[key] for item in sequence]
                episode_group.create_dataset(key, data=np.asarray(values))

    def _resolve_state_path(self, dataset_id: str) -> Path:
        base = minari_root()
        return base / dataset_id / "data" / "state_data.hdf5"
