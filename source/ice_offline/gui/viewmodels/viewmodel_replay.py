from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ice_offline.gui.models.model_replay import EpisodeInfo


@dataclass(frozen=True)
class ReplayState:
    select_title: str
    select_labels: list[str]
    select_index: int
    slider_value: int
    slider_max: int
    step_jump: int
    frame: np.ndarray | None


class ReplayViewModel:
    # ====================
    # Init
    # ====================
    def __init__(self, model) -> None:
        self._model = model
        self._labels: list[str] = []
        self._episodes: list[EpisodeInfo] = []
        self._all_mapping: list[tuple[int, int]] = []

        self._all_mode = True
        self._selected_episode = 0
        self._selected_step = 0
        self._max_step = 0
        self._step_jump = 1

    def initial_state(self) -> ReplayState:
        return self._state()

    # ====================
    # Public API
    # ====================
    def load_file(self, path: str) -> ReplayState:
        path_obj = Path(path)
        if path_obj.name == "main_data.hdf5":
            return self.load_minari(path)
        if path_obj.name == "d4rl_data.hdf5":
            return self.load_d4rl(path)
        raise FileNotFoundError(f"unsupported dataset file: {path}")

    def load_minari(self, path: str) -> ReplayState:
        self._model.load_minari(path)
        self._episodes = self._model.episodes()
        self._labels = ["[all]"] + [f"episode_{episode.id}" for episode in self._episodes]
        self._all_mapping = self._build_all_mapping()
        return self.set_episode(0)

    def load_d4rl(self, path: str) -> ReplayState:
        self._model.load_d4rl(path)
        self._episodes = self._model.episodes()
        self._labels = ["[all]"] + [f"episode_{episode.id}" for episode in self._episodes]
        self._all_mapping = self._build_all_mapping()
        return self.set_episode(0)

    def load_folder(self, path: str) -> ReplayState:
        main_data_path = self._model.scan_file(path, "main_data.hdf5")
        if main_data_path:
            return self.load_minari(main_data_path)

        d4rl_data_path = self._model.scan_file(path, "d4rl_data.hdf5")
        if d4rl_data_path:
            return self.load_d4rl(d4rl_data_path)

        raise FileNotFoundError(f"missing main_data.hdf5 or d4rl_data.hdf5 under: {path}")

    def set_episode(self, index: int) -> ReplayState:
        if not self._episodes:
            return self._state()

        index = max(0, min(index, len(self._episodes)))
        self._selected_episode = index
        self._all_mode = index == 0

        if self._all_mode:
            self._max_step = len(self._all_mapping) - 1
        else:
            self._max_step = self._episodes[index - 1].steps - 1

        return self.set_step(0)

    def set_step(self, value: int) -> ReplayState:
        value = max(0, min(value, self._max_step))
        self._selected_step = value
        return self._state()

    def set_step_jump(self, value: int) -> ReplayState:
        self._step_jump = max(1, int(value))
        return self._state()

    def step_jump(self) -> int:
        return self._step_jump

    def close(self) -> None:
        self._model.close()

    # ====================
    # Internal Helpers
    # ====================
    def _state(self) -> ReplayState:
        frame = self._render()
        return ReplayState(
            select_title="Episode",
            select_labels=self._labels,
            select_index=self._selected_episode,
            slider_value=self._selected_step,
            slider_max=self._max_step,
            step_jump=self._step_jump,
            frame=frame,
        )

    def _render(self) -> np.ndarray | None:
        if not self._episodes:
            return None

        if self._all_mode:
            episode, step = self._all_mapping[self._selected_step]
        else:
            episode = self._episodes[self._selected_episode - 1].id
            step = self._selected_step

        return self._model.render(episode, step)

    def _build_all_mapping(self) -> list[tuple[int, int]]:
        mapping: list[tuple[int, int]] = []
        for episode in self._episodes:
            for step in range(episode.steps):
                mapping.append((episode.id, step))
        return mapping
