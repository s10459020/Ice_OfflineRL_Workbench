from __future__ import annotations

import math
from types import MethodType

import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.utils.rendering import fill_coords, highlight_img, point_in_triangle, rotate_fn

from .render_delay_wrapper import RenderDelayWrapper


class TrailDelayWrapper(gym.Wrapper):
    """Store trail states (agent position/direction) for later rendering."""

    def __init__(self, env: gym.Env) -> None:
        if not isinstance(env, RenderDelayWrapper):
            raise TypeError("TrailDelayWrapper requires env to be RenderDelayWrapper.")
        super().__init__(env)
        self._base_env = self.env.unwrapped
        self._patch_grid_render()
        # Each item: ((x, y), agent_dir)
        self.trails: list[tuple[tuple[int, int], int]] = []
        self._trail_color = np.array([70, 160, 255], dtype=np.float32)
        self._arrow_masks_cache: dict[int, dict[int, np.ndarray]] = {}
        # Provided by outer RenderDelayWrapper.
        self._last_seen_render_tick: int = 0

    def _patch_grid_render(self) -> None:
        grid = self._base_env.grid
        grid.render = MethodType(self._trail_render, grid)

    def _trail_render(
        self,
        grid_self: Grid,
        tile_size: int,
        agent_pos: tuple[int, int],
        agent_dir: int | None = None,
        highlight_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(grid_self.width, grid_self.height), dtype=bool)

        trails_by_cell: dict[tuple[int, int], list[tuple[int, int]]] = {}
        trail_den = max(1, len(self.trails) - 1)
        for idx, (pos, trail_dir) in enumerate(self.trails):
            trails_by_cell.setdefault(pos, []).append((idx, trail_dir))

        arrow_masks = self._get_arrow_masks(tile_size)

        width_px = grid_self.width * tile_size
        height_px = grid_self.height * tile_size
        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        for j in range(0, grid_self.height):
            for i in range(0, grid_self.width):
                cell = grid_self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=None,
                    highlight=False,
                    tile_size=tile_size,
                )

                trail_overlay_rgb = np.zeros_like(tile_img, dtype=np.float32)
                trail_overlay_alpha = np.zeros((tile_size, tile_size), dtype=np.float32)

                for idx, trail_dir in trails_by_cell.get((i, j), []):
                    alpha = 0.10 + 0.60 * (idx / trail_den)
                    mask = arrow_masks[int(trail_dir)]

                    # Trails overwrite each other on this overlay.
                    trail_overlay_rgb[mask] = self._trail_color
                    trail_overlay_alpha[mask] = alpha

                # Blend trail overlay to base tile exactly once.
                alpha_map = trail_overlay_alpha[..., None]
                tile_img = (1.0 - alpha_map) * tile_img + alpha_map * trail_overlay_rgb

                if agent_here and agent_dir is not None:
                    tri_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
                    tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
                    fill_coords(tile_img, tri_fn, (255, 0, 0))

                if highlight_mask[i, j]:
                    highlight_img(tile_img)

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

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
        # MiniGrid recreates grid on reset; re-patch the new instance method.
        self._patch_grid_render()
        self.trails.clear()
        self.trails.append(self._capture_agent_state())
        self._last_seen_render_tick = self.env.render_tick
        return out

    def step(self, action):
        out = self.env.step(action)
        self.trails.append(self._capture_agent_state())
        return out

    def render(self):
        out = self.env.render()
        if self.env.render_tick > self._last_seen_render_tick:
            self.trails.clear()
            self._last_seen_render_tick = self.env.render_tick
        return out
