from typing import Any, Protocol, Type

from ice_offline.dataset._spec import Dataset
from ice_offline.dataset._types import Episode
from ice_offline.store.state._spec import State
from ice_offline.store.state.op_dataset import StateDataset


class BaseConverter(Protocol):
    def convert_episode(self, trajectory: Episode) -> list[State]:
        ...


class StateConverter:
    # ====================
    # Init
    # ====================
    def __init__(self, dataset: Dataset, converter_cls: Type[BaseConverter]) -> None:
        self._dataset = dataset
        self._converter = converter_cls()

    # ====================
    # Public API
    # ====================
    def convert(self) -> StateDataset:
        total = int(self._dataset.episode_count)
        
        print(f"[state-convert] start episodes={total}")
        episodes: list[list[dict[str, Any]]] = []
        state_cls: Type[State] = State
        for episode_index in range(total):
            trajectory = self._dataset.episodes[episode_index]
            states = self._converter.convert_episode(trajectory)
            state_cls = type(states[0])
            episodes.append([state.serialize() for state in states])
            if (episode_index + 1) % 100 == 0 or (episode_index + 1) == total:
                print(f"[state-convert] converted {episode_index + 1}/{total}")

        print("[state-convert] writing state_data.hdf5 ...")
        return StateDataset.write(
            path=self._dataset.path.with_name("state_data.hdf5"),
            state_cls=state_cls,
            episodes=episodes,
        )
