"""Episode metadata model for GUI views and presenters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeInfo:
    episode_id: int
    step_count: int
