import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ice_offline.config.datasets import DatasetEntry
from ice_offline.config.datasets import source_dataset_entries
from ice_offline.dataset._spec import Dataset
from ice_offline.gui.models.model_replay import EpisodeInfo


@dataclass(frozen=True)
class ReplayLoadedState:
    select_title: str
    select_labels: list[str]
    select_index: int
    slider_value: int
    slider_max: int
    step_jump: int
    frame: np.ndarray | None


@dataclass(frozen=True)
class ReplayEpisodeState:
    select_index: int
    slider_value: int
    slider_max: int
    frame: np.ndarray | None


@dataclass(frozen=True)
class ReplayFrameState:
    slider_value: int
    frame: np.ndarray | None


class ReplayViewModel:
    # ====================
    # Init
    # ====================
    def __init__(self, model) -> None:
        self._model = model
        self._datasets = source_dataset_entries()
        self._labels: list[str] = []
        self._episodes: list[EpisodeInfo] = []
        self._all_mapping: list[tuple[int, int]] = []

        self._all_mode = True
        self._selected_episode = 0
        self._selected_step = 0
        self._max_step = 0
        self._step_jump = 1

    def initial_state(self) -> ReplayLoadedState:
        return self._loaded_view_state()

    # ====================
    # Public API
    # ====================
    def datasets(self) -> list[DatasetEntry]:
        return self._datasets

    def load_dataset(self, dataset_cls: type[Dataset]) -> ReplayLoadedState:
        self._model.load_dataset(dataset_cls())
        return self._loaded_state()

    def load_run_data(self, path: str) -> ReplayLoadedState:
        metadata_path = self._model.scan_file(path, "metadata.json")
        if metadata_path is None:
            raise FileNotFoundError(f"missing metadata.json under: {path}")
        data_dir = Path(metadata_path).parent
        data_path = data_dir / "main_data.hdf5"
        if not data_path.exists():
            data_path = data_dir / "eval_data.hdf5"
        if not data_path.exists():
            raise FileNotFoundError(f"missing main_data.hdf5 or eval_data.hdf5 under: {data_dir}")
        self._model.load_minari(str(data_path))
        return self._loaded_state()

    def set_episode(self, index: int) -> ReplayEpisodeState:
        if not self._episodes:
            return self._episode_state()

        index = max(0, min(index, len(self._episodes)))
        self._selected_episode = index
        self._all_mode = index == 0

        if self._all_mode:
            self._max_step = len(self._all_mapping) - 1
        else:
            self._max_step = self._episodes[index - 1].steps - 1

        self._selected_step = 0
        return self._episode_state()

    def set_step(self, value: int) -> ReplayFrameState:
        value = max(0, min(value, self._max_step))
        self._selected_step = value
        return self._frame_state()

    def set_step_jump(self, value: int) -> None:
        self._step_jump = max(1, int(value))

    def step_jump(self) -> int:
        return self._step_jump

    def close(self) -> None:
        self._model.close()

    # ====================
    # Internal Helpers
    # ====================
    def _loaded_view_state(self) -> ReplayLoadedState:
        return ReplayLoadedState(
            select_title="Episode",
            select_labels=self._labels,
            select_index=self._selected_episode,
            slider_value=self._selected_step,
            slider_max=self._max_step,
            step_jump=self._step_jump,
            frame=self._render(),
        )

    def _episode_state(self) -> ReplayEpisodeState:
        return ReplayEpisodeState(
            select_index=self._selected_episode,
            slider_value=self._selected_step,
            slider_max=self._max_step,
            frame=self._render(),
        )

    def _frame_state(self) -> ReplayFrameState:
        return ReplayFrameState(
            slider_value=self._selected_step,
            frame=self._render(),
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

    def _loaded_state(self) -> ReplayLoadedState:
        self._episodes = self._model.episodes()
        self._labels = ["[all]"] + [f"episode_{episode.id}" for episode in self._episodes]
        self._all_mapping = self._build_all_mapping()
        self._selected_episode = 0
        self._selected_step = 0
        self._all_mode = True
        self._max_step = max(0, len(self._all_mapping) - 1)
        return self._loaded_view_state()

    def _read_metadata(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
