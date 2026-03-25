"""Minari-backed dataset service for GUI startup data loading."""

from __future__ import annotations

import minari
import numpy as np

from ice_offline.env.model import EpisodeInfo
from ice_offline.env.model import State
from ice_offline.env.replay.state_io_wrapper import ensure_state_io


class MinariDatasetService:
    """Loads episode metadata from a Minari dataset."""

    def __init__(self, dataset_id: str) -> None:
        self._dataset = minari.load_dataset(dataset_id)
        self._env = self._dataset.recover_environment(render_mode="rgb_array")
        self._env = ensure_state_io(self._env)
        # OrderEnforcer requires at least one reset before render.
        self._env.reset()
        self._set_state = self._env.get_wrapper_attr("set_state")

    def list_episodes(self) -> list[EpisodeInfo]:
        episodes: list[EpisodeInfo] = []
        for idx, trajectory in enumerate(self._dataset.iterate_episodes()):
            episodes.append(EpisodeInfo(episode_id=idx, step_count=len(trajectory.rewards) + 1))
        return episodes

    def render_episode_step(self, episode_id: int, step_index: int) -> np.ndarray:
        trajectory = self._dataset[episode_id]
        state_payload = trajectory.infos["state"]
        payload = {key: state_payload[key][step_index] for key in state_payload}
        state = State.from_serialized(payload)
        self._set_state(state)
        return self._env.render()

    def close(self) -> None:
        self._env.close()
