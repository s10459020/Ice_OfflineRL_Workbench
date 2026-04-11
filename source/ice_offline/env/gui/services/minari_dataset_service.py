"""Minari-backed dataset service for GUI startup data loading."""

from __future__ import annotations

import numpy as np

from ice_offline.data import EpisodeInfo
from ice_offline.env.visualization import BasicUnit, DistributionUnit, RenderLayer
from ice_offline.env.visualization.overlay_loader import OverlayLoader
from ice_offline.env.visualization.unit_trail import TrailUnit


class MinariDatasetService:
    """Loads episode metadata from a Minari dataset."""

    def __init__(self, dataset_id: str, distribution_style: str = "ring") -> None:
        self._loader = OverlayLoader(
            dataset_id,
            units=[BasicUnit(), TrailUnit(), DistributionUnit(style=distribution_style)],
            #units=[BasicUnit(), TrailUnit(), DistributionUnit(quantize_mode="fixed")],
            render_mode="rgb_array",
        )
        self._dataset = self._loader.get_dataset()
        self._loaded_episode_id: int | None = None

    def list_episodes(self) -> list[EpisodeInfo]:
        episodes: list[EpisodeInfo] = []
        for idx, trajectory in enumerate(self._dataset.iterate_episodes()):
            episodes.append(EpisodeInfo(episode_id=idx, step_count=len(trajectory.rewards) + 1))
        return episodes

    def render_episode_step(self, episode_id: int, step_index: int) -> np.ndarray:
        if self._loaded_episode_id != episode_id:
            self._loader.load(episode_id)
            self._loaded_episode_id = episode_id
        self._loader.seek(step_index)
        return self._loader.render()

    def set_trail_enabled(self, enabled: bool) -> None:
        self._loader.engine.set_enabled(RenderLayer.TRAIL, enabled)

    def set_distribution_enabled(self, enabled: bool) -> None:
        self._loader.engine.set_enabled(RenderLayer.DISTRIBUTION, enabled)

    def close(self) -> None:
        self._loader.close()
