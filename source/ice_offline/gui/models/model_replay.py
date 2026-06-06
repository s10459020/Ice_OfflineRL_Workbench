from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

from ice_offline.dataset._spec import Dataset
from ice_offline.dataset.loader.d4rl.loader import D4rlLoader
from ice_offline.dataset.loader.minari.loader import MinariLoader
from ice_offline.store.state._lookup import STATE_OPS
from ice_offline.store.state._spec import StateIO
from ice_offline.store.state.op_converter import StateConverter
from ice_offline.store.state.op_dataset import StateDataset


@dataclass(frozen=True)
class EpisodeInfo:
    id: int
    steps: int


class ReplayModel:
    def __init__(self) -> None:
        self._env: gym.Env | None = None
        self._episodes: list[EpisodeInfo] = []
        self._state_io: StateIO | None = None
        self._state_dataset: StateDataset | None = None

    def load_minari(self, path: str) -> None:
        self.close()

        data_path = Path(path)
        print(f"loading dataset: {data_path}")
        dataset = Dataset(path=data_path, loader=MinariLoader(data_path))
        self.load_dataset(dataset)

    def load_d4rl(self, path: str) -> None:
        self.close()

        data_path = Path(path)
        print(f"loading dataset: {data_path}")
        dataset = Dataset(path=data_path, loader=D4rlLoader(data_path))
        self.load_dataset(dataset)

    def load_dataset(self, dataset: Dataset) -> None:
        self.close()
        if dataset.env_id not in STATE_OPS:
            raise ValueError(f"unsupported replay state env: {dataset.env_id}")
        state_cls, state_io_cls, state_converter_cls = STATE_OPS[dataset.env_id]
        
        state_data_path = dataset.path.with_name("state_data.hdf5")
        if not state_data_path.exists():
            print(f"converting state data: {state_data_path}")
            converter = StateConverter(dataset=dataset, converter_cls=state_converter_cls)
            converter.convert()

        print(f"creating env: {dataset.env_id}")
        self._env = gym.make(dataset.env_id, render_mode="rgb_array")
        self._env.reset()
        self._state_io = state_io_cls(self._env)

        print(f"loading state dataset: {state_data_path}")
        self._state_dataset = StateDataset.load_dataset(path=state_data_path, state_cls=state_cls)
        self._episodes = [
            EpisodeInfo(id=i, steps=self._state_dataset.step_counts[i] + 1)
            for i in range(self._state_dataset.episode_count)
        ]

    def scan_file(self, path: str, name: str) -> str | None:
        stack = [path]
        while stack:
            current = Path(stack.pop())
            for child in sorted(current.iterdir(), key=lambda item: item.name, reverse=True):
                if child.is_dir():
                    stack.append(child)
                elif child.name == name:
                    return str(child)

        return None

    def episodes(self) -> list[EpisodeInfo]:
        return self._episodes

    def render(self, episode: int, step: int) -> np.ndarray | None:
        state = self._state_dataset.read_step(episode, step)
        self._state_io.set_state(state)
        return self._env.render()

    def close(self) -> None:
        if self._env:
            self._env.close()
        if self._state_dataset:
            self._state_dataset.close()
