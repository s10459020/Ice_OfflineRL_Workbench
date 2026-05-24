from typing import Any, Type

from ice_offline.pipeline.state._spec import State
from ice_offline.pipeline.state_operator.state_dataset import StateDataset


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
        episodes: list[list[dict[str, Any]]] = []
        for episode_index in range(self._dataset.total_episodes):
            trajectory = self._dataset[episode_index]
            states = self._converter.convert_episode(trajectory)
            episodes.append([state.serialize() for state in states])
        return StateDataset.write(
            dataset_id=self._dataset.spec.dataset_id,
            state_cls=State,
            episodes=episodes,
        )
