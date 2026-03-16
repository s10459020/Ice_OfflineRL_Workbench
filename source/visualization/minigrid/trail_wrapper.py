import math
import numpy as np
import gymnasium as gym

from minigrid.utils.rendering import fill_coords, point_in_triangle, rotate_fn

from .render_overlay_wrapper import OverlayDependentWrapper


class TrailWrapper(OverlayDependentWrapper):
    """Render a trail overlay using recent (position, direction) states."""

    def __init__(
        self,
        env: gym.Env,
        max_trails: int = 64,
        trail_layer: int = 30,
        clear_on_render: bool = False,
    ) -> None:
        if max_trails < 1:
            raise ValueError("max_trails must be >= 1")
        
        super().__init__(
            env,
            overlay_fn=self._overlay_trail,
            overlay_layer=trail_layer,
        )

        # Trail state.s
        # Ordered history: ((x, y), direction)
        # Per-cell lookup built before each render: (x, y) -> [(idx, dir), ...]
        self._trails: list[tuple[tuple[int, int], int]] = []
        self._trails_by_cell: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._max_trails = max_trails

        # Trail rendering style.
        self._trail_color = np.array([70, 160, 255], dtype=np.float32)

        # Render-time caches.
        self._arrow_masks_cache: dict[int, dict[int, np.ndarray]] = {}

        self._clear_on_render = clear_on_render

    # ------------------------------------------------------------------
    # Env lifecycle
    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._trails.clear()
        self._push_current_state()
        return out

    def step(self, action):
        out = self.env.step(action)
        self._push_current_state()
        return out

    def render(self):
        self._rebuild_trails_by_cell()
        out = self.env.render()

        if self._clear_on_render:
            self._trails.clear()

        return out

    # ------------------------------------------------------------------
    # Overlay callback
    # ------------------------------------------------------------------
    def _overlay_trail(self, tile_img: np.ndarray, ctx: dict[str, object]) -> None:
        i = ctx["i"]
        j = ctx["j"]
        tile_size = ctx["tile_size"]
        entries = self._trails_by_cell.get((i, j))
        if not entries:
            return

        trail_den = max(1, len(self._trails) - 1)
        arrow_masks = self._get_arrow_masks(tile_size)
        overlay_rgb = np.zeros_like(tile_img, dtype=np.float32)
        overlay_alpha = np.zeros((tile_size, tile_size), dtype=np.float32)

        for idx, trail_dir in entries:
            alpha = 0.10 + 0.60 * (idx / trail_den)
            mask = arrow_masks[trail_dir]
            overlay_rgb[mask] = self._trail_color
            overlay_alpha[mask] = alpha

        alpha_map = overlay_alpha[..., None]
        blended = (1.0 - alpha_map) * tile_img.astype(np.float32) + alpha_map * overlay_rgb
        tile_img[:, :, :] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _capture_agent_state(self) -> tuple[tuple[int, int], int]:
        base = self.env.unwrapped
        x, y = base.agent_pos
        return (x, y), base.agent_dir

    def _rebuild_trails_by_cell(self) -> None:
        trails_by_cell: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for idx, (pos, trail_dir) in enumerate(self._trails):
            trails_by_cell.setdefault(pos, []).append((idx, trail_dir))
        self._trails_by_cell = trails_by_cell

    def _push_current_state(self) -> None:
        self._trails.append(self._capture_agent_state())
        overflow = len(self._trails) - self._max_trails
        if overflow > 0:
            del self._trails[:overflow]

    def _get_arrow_masks(self, tile_size: int) -> dict[int, np.ndarray]:
        arrow_masks = self._arrow_masks_cache.get(tile_size)
        if arrow_masks is not None:
            return arrow_masks

        arrow_masks = {}
        for d in range(4):
            ghost = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            tri_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * d)
            fill_coords(ghost, tri_fn, (255, 255, 255))
            arrow_masks[d] = ghost[:, :, 0] > 0

        self._arrow_masks_cache[tile_size] = arrow_masks
        return arrow_masks
