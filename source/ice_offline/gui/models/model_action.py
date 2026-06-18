from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ice_offline.gui.models.model_replay import ReplayModel
from ice_offline.store.minari.loader import MinariLoader
from ice_offline.store.probe.op_dataset import ProbeDataset


@dataclass(frozen=True)
class ActionCurve:
    label: str
    xs: np.ndarray
    ys: np.ndarray
    action_value: float
    source_min: float
    source_max: float


@dataclass(frozen=True)
class ActionEpisodeInfo:
    id: int
    steps: int


class ActionModel:
    def __init__(self) -> None:
        self._replay = ReplayModel()
        self._actions_main: list[np.ndarray] = []
        self._episodes_1: list[list[dict[str, np.ndarray]]] = []
        self._episodes_2: list[list[dict[str, np.ndarray]]] = []

    def load_run_data(self, path: str, index: int) -> None:
        data_path = Path(path)
        probe_path = self.scan_file(str(data_path), "probe_data.hdf5")
        if probe_path is None:
            raise FileNotFoundError(f"missing probe_data.hdf5 under: {path}")
        main_path = self.scan_file(str(data_path), "main_data.hdf5")
        if main_path is None:
            raise FileNotFoundError(f"missing main_data.hdf5 under: {path}")
        dataset = ProbeDataset.load_dataset(Path(probe_path))
        try:
            episodes = dataset.load_episodes()
        finally:
            dataset.close()
        actions = [episode.actions for episode in MinariLoader(Path(main_path), device="cpu").load_episodes()]
        if index == 1:
            self._replay.load_minari(str(main_path))
            self._actions_main = actions
            self._episodes_1 = episodes
            return
        self._episodes_2 = episodes

    def scan_file(self, path: str, name: str) -> str | None:
        stack = [path]
        while stack:
            current = Path(stack.pop())
            for child in sorted(current.iterdir(), key=lambda item: item.name, reverse=True):
                if child.is_dir():
                    stack.append(str(child))
                elif child.name == name:
                    return str(child)
        return None

    def episodes(self) -> list[ActionEpisodeInfo]:
        return [ActionEpisodeInfo(id=index, steps=len(episode)) for index, episode in enumerate(self._episodes_1)]

    def curves(self, episode: int, step: int, index: int) -> list[ActionCurve]:
        episodes = self._episodes_1 if index == 1 else self._episodes_2
        if not episodes:
            return []
        payload = episodes[episode][step]
        actions = np.asarray(payload["actions"], dtype=np.float32)
        values = np.asarray(payload["values"], dtype=np.float32)
        flat_actions = actions.reshape(actions.shape[0], -1)
        action_dim = flat_actions.shape[1]
        sample_count = flat_actions.shape[0] // action_dim

        curves: list[ActionCurve] = []
        for dim in range(action_dim):
            start = dim * sample_count
            end = start + sample_count
            segment_values = values[start:end].copy()
            source_min = float(segment_values.min())
            source_max = float(segment_values.max())
            if source_min == source_max:
                normalized_values = np.zeros_like(segment_values, dtype=np.float32)
            else:
                normalized_values = ((segment_values - source_min) / (source_max - source_min)).astype(np.float32)
            curves.append(
                ActionCurve(
                    label=f"a{dim}",
                    xs=flat_actions[start:end, dim].copy(),
                    ys=normalized_values,
                    action_value=float(self._actions_main[episode][step].reshape(-1)[dim]),
                    source_min=source_min,
                    source_max=source_max,
                )
            )
        return curves

    def render(self, episode: int, step: int) -> np.ndarray | None:
        return self._replay.render(episode, step)

    def close(self) -> None:
        self._replay.close()
        self._actions_main = []
        self._episodes_1 = []
        self._episodes_2 = []
