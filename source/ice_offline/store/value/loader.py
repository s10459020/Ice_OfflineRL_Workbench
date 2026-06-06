from pathlib import Path

import h5py
import numpy as np
from ice_offline.config.paths import DATASETS_ROOT


class ValueLoader:
    """Load value_data.hdf5 from Minari dataset folder."""

    # ====================
    # init
    # ====================
    def __init__(
        self,
        dataset_id: str,
        data_file_name: str = "value_data.hdf5",
        dataset_key: str = "values",
    ) -> None:
        self._path = self._resolve_data_path(dataset_id, data_file_name)
        self._dataset_key = dataset_key
        self._h5 = h5py.File(self._path, "r")
        self._episode_keys = self._list_episode_keys()

    # ====================
    # public
    # ====================
    def close(self) -> None:
        self._h5.close()

    def get_episode_count(self) -> int:
        return len(self._episode_keys)

    def load_episode(self, episode_index: int) -> list[np.ndarray]:
        data = self._h5[self._episode_keys[episode_index]][self._dataset_key][()]
        return [data[step_index] for step_index in range(data.shape[0])]

    def load_step(self, episode_index: int, step_index: int) -> np.ndarray:
        data = self._h5[self._episode_keys[episode_index]][self._dataset_key]
        return data[step_index]

    # ====================
    # helping
    # ====================
    def _resolve_data_path(self, dataset_id: str, data_file_name: str) -> Path:
        return DATASETS_ROOT / dataset_id / "data" / data_file_name

    def _list_episode_keys(self) -> list[str]:
        keys = [key for key in self._h5.keys() if key.startswith("episode_")]
        return sorted(keys, key=lambda key: int(key.split("_")[1]))


