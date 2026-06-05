from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

from ice_offline.dataset._spec import Dataset
from ice_offline.store.state._spec import State, StateIO
from ice_offline.store.state.op_converter import StateConverter
from ice_offline.store.state.op_dataset import StateDataset


@dataclass(frozen=True)
class EpisodeInfo:
    id: int
    steps: int


class ReplayModel:
    def __init__(self, state_cls: type[State], state_io_cls: type[StateIO], state_converter_cls: type | None = None) -> None:
        self._env: gym.Env | None = None
        self._episodes: list[EpisodeInfo] = []
        self._state_io: StateIO | None = None
        self._state_dataset: StateDataset | None = None
        self._state_cls: type[State] = state_cls
        self._state_io_cls: type[StateIO] = state_io_cls
        self._state_converter_cls: type | None = state_converter_cls

    def load_dataset(self, path: str) -> None:
        print(f"[replay] load start path={path}")
        self.close()

        main_data_path = Path(path)
        print("[replay] loading minari buffer ...")
        dataset = Dataset(id=main_data_path.parent.parent.name, path=main_data_path)
        print("[replay] minari loaded")

        state_data_path = main_data_path.with_name("state_data.hdf5")
        if not state_data_path.exists():
            print(f"[replay] missing state file -> convert: {state_data_path}")
            converter = StateConverter(dataset=dataset, converter_cls=self._state_converter_cls)
            converter.convert()
            print("[replay] state convert done")
        else:
            print(f"[replay] state file found: {state_data_path}")

        env_id = dataset.env_id
        print(f"[replay] creating env: {env_id}")
        self._env = gym.make(env_id, render_mode="rgb_array")
        self._env.reset()
        self._state_io = self._state_io_cls(self._env)

        print("[replay] loading state dataset ...")
        self._state_dataset = StateDataset.load_dataset(path=state_data_path, state_cls=self._state_cls)
        print(f"[replay] state dataset loaded episodes={self._state_dataset.episode_count}")
        self._episodes = [
            EpisodeInfo(id=i, steps=self._state_dataset.step_counts[i] + 1)
            for i in range(self._state_dataset.episode_count)
        ]
        print("[replay] load done")

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
