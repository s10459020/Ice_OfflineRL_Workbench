from pathlib import Path
from typing import Any, Type

import h5py
import minari
import numpy as np

from ice_offline.pipeline.state._spec import State
from ice_offline.tools.paths import minari_root


class StateConverter:
    # ====================
    # Init
    # ====================
    def __init__(self, dataset_id: str, converter_cls: Type) -> None:
        self._dataset = minari.load_dataset(dataset_id)
        self._path = self._resolve_state_path(dataset_id)
        self._converter = converter_cls()

    # ====================
    # Public API
    # ====================
    def convert(self) -> Path:
        state_episodes: list[list[dict[str, Any]]] = []
        for trajectory in self._dataset.iterate_episodes():
            states = self._converter.convert_episode(trajectory)
            state_episodes.append([state.serialize() for state in states])

        return self._save_state_data(state_episodes)

    # ====================
    # Private
    # ====================
    def _save_state_data(self, episodes: list[list[dict[str, Any]]]) -> Path:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(self._path, "w") as h5_file:
            for ep_idx, seq in enumerate(episodes):
                ep_group = h5_file.require_group(f"episode_{ep_idx}")
                keys = seq[0].keys()
                for key in keys:
                    values = [item[key] for item in seq]
                    ep_group.create_dataset(key, data=np.asarray(values))
        return self._path

    def _resolve_state_path(self, dataset_id: str) -> Path:
        base = minari_root()
        return base / dataset_id / "data" / "state_data.hdf5"
