from dataclasses import dataclass

import numpy as np

from ice_offline.gui.models.model_action import ActionCurve
from ice_offline.gui.models.model_action import ActionEpisodeInfo


@dataclass(frozen=True)
class ActionLoadedState:
    select_title: str
    select_labels: list[str]
    select_index: int
    slider_value: int
    slider_max: int
    x_name: str
    y_name: str
    curves_1: list[ActionCurve]
    curves_2: list[ActionCurve]
    frame: np.ndarray | None


@dataclass(frozen=True)
class ActionEpisodeState:
    select_index: int
    slider_value: int
    slider_max: int
    curves_1: list[ActionCurve]
    curves_2: list[ActionCurve]
    frame: np.ndarray | None


@dataclass(frozen=True)
class ActionFrameState:
    slider_value: int
    curves_1: list[ActionCurve]
    curves_2: list[ActionCurve]
    frame: np.ndarray | None


class ActionViewModel:
    def __init__(self, model) -> None:
        self._model = model
        self._episodes: list[ActionEpisodeInfo] = []
        self._selected_episode = 0
        self._selected_step = 0
        self._max_step = 0
        self._x_name = "action"
        self._y_name = "log_prob"

    def initial_state(self) -> ActionLoadedState:
        return self._loaded_state_view()

    def load_run_data(self, path: str, index: int) -> ActionLoadedState:
        self._model.load_run_data(path, index)
        self._episodes = self._model.episodes()
        self._selected_episode = 0
        self._selected_step = 0
        self._max_step = max(0, self._episodes[0].steps - 1) if self._episodes else 0
        return self._loaded_state_view()

    def set_episode(self, index: int) -> ActionEpisodeState:
        if not self._episodes:
            return self._episode_state()
        index = max(0, min(index, len(self._episodes) - 1))
        self._selected_episode = index
        self._selected_step = 0
        self._max_step = max(0, self._episodes[index].steps - 1)
        return self._episode_state()

    def set_step(self, value: int) -> ActionFrameState:
        self._selected_step = max(0, min(value, self._max_step))
        return self._frame_state()

    def close(self) -> None:
        self._model.close()

    def _loaded_state_view(self) -> ActionLoadedState:
        return ActionLoadedState(
            select_title="Episode",
            select_labels=[f"episode_{episode.id}" for episode in self._episodes],
            select_index=self._selected_episode,
            slider_value=self._selected_step,
            slider_max=self._max_step,
            x_name=self._x_name,
            y_name=self._y_name,
            curves_1=self._curves(1),
            curves_2=self._curves(2),
            frame=self._render(),
        )

    def _episode_state(self) -> ActionEpisodeState:
        return ActionEpisodeState(
            select_index=self._selected_episode,
            slider_value=self._selected_step,
            slider_max=self._max_step,
            curves_1=self._curves(1),
            curves_2=self._curves(2),
            frame=self._render(),
        )

    def _frame_state(self) -> ActionFrameState:
        return ActionFrameState(
            slider_value=self._selected_step,
            curves_1=self._curves(1),
            curves_2=self._curves(2),
            frame=self._render(),
        )

    def _curves(self, index: int) -> list[ActionCurve]:
        if not self._episodes:
            return []
        return self._model.curves(self._selected_episode, self._selected_step, index)

    def _render(self) -> np.ndarray | None:
        if not self._episodes:
            return None
        return self._model.render(self._selected_episode, self._selected_step)
