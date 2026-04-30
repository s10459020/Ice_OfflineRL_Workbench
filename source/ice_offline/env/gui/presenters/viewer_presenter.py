"""MVP presenter for episode/step interaction."""


from ice_offline.data import EpisodeInfo


class ViewerPresenter:
    """Handles state transitions and prints state changes."""

    # ====================
    # Lifecycle
    # ====================
    def __init__(self, episodes: list[EpisodeInfo]) -> None:
        self._episodes = episodes
        self._selected_episode = 0
        self._all_mode = False
        self._selected_step = 0
        self._total_steps = sum(episode.step_count for episode in episodes)

    # ====================
    # Event Handlers
    # ====================
    def episode_labels(self) -> list[str]:
        return ["[all]"] + [f"episode_{episode.episode_id}" for episode in self._episodes]

    def on_episode_selected(self, episode_index: int) -> tuple[int, int, int]:
        # episode_index is from list UI: 0 => [all], 1..N => episodes[0..N-1]
        if episode_index < 0 or episode_index > len(self._episodes):
            return (0, 0, 0)
        self._selected_step = 0
        self._all_mode = (episode_index == 0)
        if self._all_mode:
            max_step = self._total_steps - 1
            self._selected_episode = 0
        else:
            self._selected_episode = episode_index - 1
            max_step = self._episodes[self._selected_episode].step_count - 1
        self._print_state("episode_changed")
        return (0, max_step, 0)

    def on_step_slider_changed(self, step: int) -> None:
        self._selected_step = step
        self._print_state("step_changed")

    def current_render_target(self) -> tuple[int, int]:
        if self._all_mode:
            return self._resolve_global_step(self._selected_step)
        return self._selected_episode, self._selected_step

    # ====================
    # Internal Helpers
    # ====================
    def _print_state(self, reason: str) -> None:
        if self._all_mode:
            episode_index, local_step = self._resolve_global_step(self._selected_step)
            episode = self._episodes[episode_index]
            print(
                f"[{reason}] episode=[all] global_step={self._selected_step}/{self._total_steps - 1} "
                f"mapped_episode=episode_{episode.episode_id} mapped_step={local_step}/{episode.step_count - 1} "
                f"total_steps={self._total_steps}"
            )
            return

        episode = self._episodes[self._selected_episode]
        print(
            f"[{reason}] episode=episode_{episode.episode_id} "
            f"step={self._selected_step}/{episode.step_count - 1} "
            f"total_steps={episode.step_count}"
        )

    def _resolve_global_step(self, global_step: int) -> tuple[int, int]:
        # Map a global [all]-mode step into (episode_index, local_step).
        offset = global_step
        for episode_index, episode in enumerate(self._episodes):
            if offset < episode.step_count:
                return episode_index, offset
            offset -= episode.step_count
        last_index = len(self._episodes) - 1
        return last_index, self._episodes[last_index].step_count - 1
