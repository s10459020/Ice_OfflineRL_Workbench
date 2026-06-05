
from pathlib import Path

import h5py
import numpy as np
from ice_offline.tools.paths import dataset_root


class OldValueLoader:
    """Load value_data.hdf5 from Minari dataset folder."""

    # ====================
    # Init
    # ====================
    def __init__(self, dataset_id: str) -> None:
        self._path = self._resolve_value_path(dataset_id)
        self._h5 = h5py.File(self._path, "r")
        self._episode_keys = self._list_episode_keys()

    # ====================
    # Public API
    # ====================
    def close(self) -> None:
        self._h5.close()

    def get_episode_count(self) -> int:
        return len(self._episode_keys)

    def load_episode(self, episode_index: int) -> list[np.ndarray]:
        data = self._h5[self._episode_keys[episode_index]]["values"][()]
        return [data[i] for i in range(data.shape[0])]

    def load_step(self, episode_index: int, step_index: int) -> np.ndarray:
        data = self._h5[self._episode_keys[episode_index]]["values"]
        return data[step_index]

    # ====================
    # Internal
    # ====================
    def _resolve_value_path(self, dataset_id: str) -> Path:
        base = dataset_root()
        return base / dataset_id / "data" / "value_data.hdf5"

    def _list_episode_keys(self) -> list[str]:
        keys = [k for k in self._h5.keys() if k.startswith("episode_")]
        return sorted(keys, key=lambda k: int(k.split("_")[1]))
