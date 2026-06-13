import h5py
import numpy as np
from pathlib import Path

from ice_offline.dataset._types import Episode
from ice_offline.store.minari.loader import MinariLoader


class EvalLoader(MinariLoader):
    def __init__(self, path: Path, device: str = "cpu") -> None:
        super().__init__(path, device)

    # ====================
    # Public API
    # ====================
    def load_batch_episodes(self) -> list[tuple[int, list[Episode]]]:
        with h5py.File(self.path, "r") as h5_file:
            batches: dict[int, list[tuple[int, Episode]]] = {}
            for key in self._episode_keys_from_h5(h5_file):
                step, index = self._episode_key_parts(key)
                episode = self._read_episode(h5_file[key])
                batches.setdefault(step, []).append((index, episode))

        return [
            (
                step,
                [
                    episode
                    for _, episode in sorted(
                        indexed_episodes,
                        key=lambda item: item[0],
                    )
                ],
            )
            for step, indexed_episodes in sorted(batches.items())
        ]

    # ====================
    # Private Methods
    # ====================
    def _episode_keys_from_h5(self, file: h5py.File) -> list[str]:
        return sorted(
            [
                key
                for key in file.keys()
                if key.startswith("episode_")
            ],
            key=lambda key: tuple(
                map(int, key.split("_")[1:])
            ),
        )

    def _episode_key_parts(self, key: str) -> tuple[int, int]:
        parts = key.split("_")
        return int(parts[1]), int(parts[2])

    def _read_episode(self, episode) -> Episode:
        return Episode(
            observations=self._read_node(episode["observations"]),
            actions=np.asarray(episode["actions"]),
            rewards=np.asarray(episode["rewards"]),
            terminations=np.asarray(episode["terminations"]),
            truncations=np.asarray(episode["truncations"]),
            infos=None,
        )
