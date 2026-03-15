from __future__ import annotations

import gymnasium as gym
import numpy as np
from minigrid.utils.rendering import fill_coords, point_in_triangle, rotate_fn

from .render_delay_wrapper import RenderDelayWrapper
from .render_overlay_wrapper import RenderLayer, RenderOverlayWrapper


class TrailDelayWrapper(gym.Wrapper):
    """Store trail states (agent position/direction) for later rendering."""

    def __init__(self, env: gym.Env) -> None:
        if not isinstance(env, RenderOverlayWrapper):
            raise TypeError("TrailDelayWrapper requires env to be RenderOverlayWrapper.")
        if not isinstance(env.env, RenderDelayWrapper):
            raise TypeError("TrailDelayWrapper requires RenderOverlayWrapper(RenderDelayWrapper(env)).")
        super().__init__(env)
        # Each item: ((x, y), agent_dir)
        self.trails: list[tuple[tuple[int, int], int]] = []
        self._trail_color = np.array([70, 160, 255], dtype=np.float32)
        self._arrow_masks_cache: dict[int, dict[int, np.ndarray]] = {}
        self._trails_by_cell: dict[tuple[int, int], list[tuple[int, int]]] = {}
        # Provided by inner RenderDelayWrapper.
        self._last_seen_render_tick: int = 0
        self.env.register_overlay(
            name="trail",
            fn=self._overlay_trail,
            layer=int(RenderLayer.TRAIL),
            z=0,
            enabled=True,
        )

    def _overlay_trail(self, tile_img: np.ndarray, ctx: dict[str, object]) -> None:
        i = int(ctx["i"])
        j = int(ctx["j"])
        tile_size = int(ctx["tile_size"])
        trail_den = max(1, len(self.trails) - 1)
        entries = self._trails_by_cell.get((i, j))
        if not entries:
            return

        arrow_masks = self._get_arrow_masks(tile_size)
        trail_overlay_rgb = np.zeros_like(tile_img, dtype=np.float32)
        trail_overlay_alpha = np.zeros((tile_size, tile_size), dtype=np.float32)

        for idx, trail_dir in entries:
            alpha = 0.10 + 0.60 * (idx / trail_den)
            mask = arrow_masks[int(trail_dir)]
            trail_overlay_rgb[mask] = self._trail_color
            trail_overlay_alpha[mask] = alpha

        alpha_map = trail_overlay_alpha[..., None]
        blended = (1.0 - alpha_map) * tile_img.astype(np.float32) + alpha_map * trail_overlay_rgb
        tile_img[:, :, :] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

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

    def _capture_agent_state(self) -> tuple[tuple[int, int], int]:
        base = self.env.unwrapped
        x, y = base.agent_pos
        return (int(x), int(y)), int(base.agent_dir)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self.trails.clear()
        self.trails.append(self._capture_agent_state())
        self._last_seen_render_tick = self.env.render_tick
        return out

    def step(self, action):
        out = self.env.step(action)
        self.trails.append(self._capture_agent_state())
        return out

    def render(self):
        trails_by_cell: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for idx, (pos, trail_dir) in enumerate(self.trails):
            trails_by_cell.setdefault(pos, []).append((idx, trail_dir))
        self._trails_by_cell = trails_by_cell

        out = self.env.render()
        if self.env.render_tick > self._last_seen_render_tick:
            self.trails.clear()
            self._last_seen_render_tick = self.env.render_tick
        return out
