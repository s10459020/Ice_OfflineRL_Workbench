"""Episode metadata model for GUI views and presenters."""


from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeInfo:
    """Episode summary info used by episode list and step slider."""

    # Dataset-level episode identifier (index-based for current services).
    episode_id: int
    # Number of available steps in this episode.
    step_count: int
