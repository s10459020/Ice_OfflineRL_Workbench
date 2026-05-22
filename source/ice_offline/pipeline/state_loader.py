
from pathlib import Path
from typing import Any

import h5py

from ice_offline.data.state import State
from ice_offline.tools.paths import minari_root


class StateLoader:
    """Load state_data.hdf5 from Minari dataset folder."""

    # ====================
    # Init
    # ====================
    def __init__(self, dataset_id: str) -> None:
        self._path = self._resolve_state_path(dataset_id)
        self._h5 = h5py.File(self._path, "r")
        self._episode_keys = self._list_episode_keys()

    # ====================
    # Public API
    # ====================
    def close(self) -> None:
        self._h5.close()

    def get_episode_count(self) -> int:
        return len(self._episode_keys)

    def load_episode(self, episode_index: int) -> list[State]:
        ep = self._h5[self._episode_keys[episode_index]]
        keys = list(ep.keys())
        values = {k: ep[k][()] for k in keys}
        length = int(values[keys[0]].shape[0])
        return [State.from_serialized({k: values[k][i] for k in keys}) for i in range(length)]

    def load_step(self, episode_index: int, step_index: int) -> State:
        ep = self._h5[self._episode_keys[episode_index]]
        payload = {k: ep[k][step_index] for k in ep.keys()}
        return State.from_serialized(payload)

    # ====================
    # Internal
    # ====================
    def _resolve_state_path(self, dataset_id: str) -> Path:
        base = minari_root()
        return base / dataset_id / "data" / "state_data.hdf5"

    def _list_episode_keys(self) -> list[str]:
        keys = [k for k in self._h5.keys() if k.startswith("episode_")]
        return sorted(keys, key=lambda k: int(k.split("_")[1]))

