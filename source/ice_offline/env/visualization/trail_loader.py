from collections.abc import Iterable
from typing import Any

from ..model.trail import Trail
from .trail_render import TrailRenderer


class TrailLoader:
    """Offline reader that fills a Trail from stored states."""

    def __init__(self, trail: Trail, *, grid_width: int, grid_height: int, tile_size: int) -> None:
        self._trail = trail
        self._grid_width = int(grid_width)
        self._grid_height = int(grid_height)
        self._tile_size = int(tile_size)

    def load_points(self, points: Iterable[tuple[tuple[int, int], int]]) -> None:
        self._trail.reset()
        self._trail.extend(points)

    def load_states(self, states: Iterable[Any]) -> None:
        points: list[tuple[tuple[int, int], int]] = []
        for state in states:
            if isinstance(state, dict):
                pos = state.get("agent_pos")
                direction = state.get("agent_dir")
            else:
                pos = getattr(state, "agent_pos", None)
                direction = getattr(state, "agent_dir", None)
            if pos is None or direction is None:
                continue
            points.append(((int(pos[0]), int(pos[1])), int(direction)))
        self.load_points(points)

    def create_renderer(
        self,
    ) -> TrailRenderer:
        return TrailRenderer(
            grid_width=self._grid_width,
            grid_height=self._grid_height,
            tile_size=self._tile_size,
        )
