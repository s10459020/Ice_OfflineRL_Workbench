from typing import Any, Type

from ice_offline.data.state._spec import State
from ice_offline.data.state.op_dataset import StateDataset


class StateConverter:
    # ====================
    # Init
    # ====================
    def __init__(self, dataset: Any, converter_cls: Type) -> None:
        self._dataset = dataset
        self._converter = converter_cls()

    # ====================
    # Public API
    # ====================
    def convert(self) -> StateDataset:
        total = int(self._dataset.total_episodes)
        print(f"[state-convert] start episodes={total}")
        episodes: list[list[dict[str, Any]]] = []
        for episode_index in range(total):
            trajectory = self._dataset[episode_index]
            states = self._converter.convert_episode(trajectory)
            episodes.append([state.serialize() for state in states])
            if (episode_index + 1) % 100 == 0 or (episode_index + 1) == total:
                print(f"[state-convert] converted {episode_index + 1}/{total}")
        print("[state-convert] writing state_data.hdf5 ...")
        return StateDataset.write(
            path=StateDataset.path(self._dataset.dataset_id),
            state_cls=State,
            episodes=episodes,
        )
