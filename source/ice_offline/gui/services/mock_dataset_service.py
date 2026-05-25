"""Mock dataset service for MVP UI interaction tests."""


from ice_offline.gui.models.episode import EpisodeInfo


class MockDatasetService:
    """Provides static episode data without env/dataset integration."""

    def list_episodes(self) -> list[EpisodeInfo]:
        return [
            EpisodeInfo(episode_id=0, step_count=5),
            EpisodeInfo(episode_id=1, step_count=10),
            EpisodeInfo(episode_id=2, step_count=15),
        ]

    def set_distribution_enabled(self, enabled: bool) -> None:
        _ = enabled
