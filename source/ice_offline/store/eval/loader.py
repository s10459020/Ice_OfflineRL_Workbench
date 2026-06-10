import h5py
import numpy as np
from pathlib import Path

from ice_offline.dataset._types import Episode
from ice_offline.dataset.loader.minari.loader import MinariLoader


class EvalLoader(MinariLoader):
    def __init__(self, path: Path, device: str = "cpu") -> None:
        super().__init__(path, device)

    # ====================
    # Loading
    # ====================
    def load_episodes(self) -> list[Episode]:
        episodes: list[Episode] = []

        with h5py.File(self.path, "r") as h5_file:
            for key in self._episode_keys_from_h5(h5_file):
                episode = h5_file[key]

                episodes.append(
                    Episode(
                        observations=self._read_node(
                            episode["observations"]
                        ),
                        actions=np.asarray(episode["actions"]),
                        rewards=np.asarray(episode["rewards"]),
                        terminations=np.asarray(
                            episode["terminations"]
                        ),
                        truncations=np.asarray(
                            episode["truncations"]
                        ),
                        infos=None,
                    )
                )

        return episodes

    # ====================
    # Public API
    # ====================
    def load(self, episode: int, index: int = 0) -> Episode:
        key = f"episode_{episode}_{index}"

        with h5py.File(self.path, "r") as h5_file:
            return self._read_episode(h5_file[key])

    def load_all(self, episode: int) -> list[Episode]:
        prefix = f"episode_{episode}_"

        with h5py.File(self.path, "r") as h5_file:
            keys = sorted(
                [
                    key
                    for key in h5_file.keys()
                    if key.startswith(prefix)
                ],
                key=lambda key: int(key.rsplit("_", 1)[1]),
            )

            return [
                self._read_episode(h5_file[key])
                for key in keys
            ]

    # ====================
    # Private Methods
    # ====================
    def _episode_keys_from_h5(self, file: h5py.File) -> list[str]:
        return sorted(file.keys(),
            key=lambda key: tuple(
                map(int, key.split("_")[1:])
            ),
        )