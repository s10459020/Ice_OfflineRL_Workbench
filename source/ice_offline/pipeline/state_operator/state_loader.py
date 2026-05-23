from pathlib import Path
from typing import Type

import h5py

from ice_offline.pipeline.state._spec import State
from ice_offline.tools.paths import minari_root


class StateLoader:
    def __init__(self, dataset_id: str, state_cls: Type[State]) -> None:
        self._path = self._resolve_state_path(dataset_id)
        self._h5 = h5py.File(self._path, "r")
        self._indices = self._read_indices()
        self._state_cls = state_cls

    # ====================
    # Public API
    # ====================
    def close(self) -> None:
        self._h5.close()

    def get_episode_count(self) -> int:
        return len(self._indices)

    def load_episode(self, episode_index: int) -> list[State]:
        ep = self._h5[self._indices[episode_index]]
        keys = list(ep.keys())
        values = {k: ep[k][()] for k in keys}
        length = int(values[keys[0]].shape[0])
        return [self._state_cls.from_serialized({k: values[k][i] for k in keys}) for i in range(length)]

    def load_step(self, episode_index: int, step_index: int) -> State:
        ep = self._h5[self._indices[episode_index]]
        payload = {k: ep[k][step_index] for k in ep.keys()}
        return self._state_cls.from_serialized(payload)

    # ====================
    # Private
    # ====================
    def _resolve_state_path(self, dataset_id: str) -> Path:
        base = minari_root()
        return base / dataset_id / "data" / "state_data.hdf5"

    def _read_indices(self) -> list[str]:
        keys = [k for k in self._h5.keys() if k.startswith("episode_")]
        return sorted(keys, key=lambda k: int(k.split("_")[1]))
