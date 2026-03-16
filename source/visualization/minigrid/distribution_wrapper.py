import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.utils.rendering import fill_coords, point_in_rect

from .render_overlay_wrapper import RenderOverlayWrapper


def minigrid_build_observation(x: int, y: int, d: int, base_env: gym.Env) -> Any:
    """Build MiniGrid observation by reusing base_env.gen_obs()."""
    old_pos = tuple(int(v) for v in base_env.agent_pos)
    old_dir = int(base_env.agent_dir)
    try:
        base_env.agent_pos = (int(x), int(y))
        base_env.agent_dir = int(d)
        return base_env.gen_obs()
    finally:
        base_env.agent_pos = old_pos
        base_env.agent_dir = old_dir


class MiniGridDirection(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


class MiniGridAction(IntEnum):
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    PICKUP = 3


class BaseRenderer(ABC):
    def __init__(
        self,
        gamma: float = 0.60,
        pickup_color: tuple[int, int, int] = (80, 220, 120),
        pickup_fill_min: float = 0.10,
    ) -> None:
        self._gamma = float(gamma)
        self._pickup_color = np.asarray(pickup_color, dtype=np.float32)
        self._pickup_fill_min = float(pickup_fill_min)

    @staticmethod
    def _flatten_directional(cell_values: np.ndarray) -> np.ndarray:
        return cell_values[:, :3].reshape(12)

    def _normalize_values(self, values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        den = vmax - vmin
        if den <= 1e-12:
            return np.zeros_like(values, dtype=np.float32)
        return np.power((values - vmin) / den, self._gamma, dtype=np.float32)

    def _normalize_scalar(self, value: float, vmin: float, vmax: float) -> float:
        den = vmax - vmin
        if den <= 1e-12:
            return 0.0
        norm = float(np.clip((value - vmin) / den, 0.0, 1.0))
        return float(np.power(norm, self._gamma))

    def _pickup_scaled_color(self, cell_values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        pickup_value = float(np.mean(cell_values[:, MiniGridAction.PICKUP]))
        pickup_norm = self._normalize_scalar(pickup_value, vmin, vmax)
        pickup_scale = self._pickup_fill_min + (1.0 - self._pickup_fill_min) * pickup_norm
        return np.clip(self._pickup_color * pickup_scale, 0.0, 255.0)

    @abstractmethod
    def render(
        self,
        tile_img_u8: np.ndarray,
        cell_values: np.ndarray,
        vmin: float,
        vmax: float,
    ) -> None:
        raise NotImplementedError


class _Rect12Renderer(BaseRenderer):
    def __init__(
        self,
        rect_color: tuple[int, int, int] = (255, 180, 60),
        rect_w_min: float = 0.015,
        rect_w_max: float = 0.135,
        rect_h: float = 0.045,
        rect_levels: int = 16,
    ) -> None:
        super().__init__()
        self._rect_color = np.asarray(rect_color, dtype=np.uint8)
        self._rect_w_min = float(rect_w_min)
        self._rect_w_max = float(rect_w_max)
        self._rect_h = float(rect_h)
        self._rect_levels = int(rect_levels)
        self._rect_mask_cache: dict[int, np.ndarray] = {}
        self._center_square_mask_cache: dict[int, np.ndarray] = {}

    def render(
        self,
        tile_img_u8: np.ndarray,
        cell_values: np.ndarray,
        vmin: float,
        vmax: float,
    ) -> None:
        norm_grid = self._normalize_values(
            self._flatten_directional(cell_values), vmin, vmax
        ).reshape(4, 3)
        masks = self._get_rect_masks(tile_img_u8.shape[0])
        action_order = (
            MiniGridAction.LEFT,
            MiniGridAction.FORWARD,
            MiniGridAction.RIGHT,
        )
        for d_idx in MiniGridDirection:
            for slot_idx, a_real in enumerate(action_order):
                level = int(round(float(norm_grid[d_idx, a_real]) * (self._rect_levels - 1)))
                level = max(1, min(self._rect_levels - 1, level))
                tile_img_u8[masks[d_idx, slot_idx, level]] = self._rect_color
        center_mask = self._get_center_square_mask(tile_img_u8.shape[0])
        tile_img_u8[center_mask] = self._pickup_scaled_color(cell_values, vmin, vmax).astype(np.uint8)

    def _get_rect_masks(self, tile_size: int) -> np.ndarray:
        masks = self._rect_mask_cache.get(tile_size)
        if masks is not None:
            return masks
        masks = np.zeros((4, 3, self._rect_levels, tile_size, tile_size), dtype=bool)
        strips: dict[MiniGridDirection, tuple[list[float], float | None, float | None]] = {
            MiniGridDirection.UP: ([0.40, 0.50, 0.60], 0.30, None),
            MiniGridDirection.DOWN: ([0.40, 0.50, 0.60], 0.70, None),
            MiniGridDirection.LEFT: ([0.40, 0.50, 0.60], None, 0.30),
            MiniGridDirection.RIGHT: ([0.40, 0.50, 0.60], None, 0.70),
        }
        for d_idx, (slots, y_anchor, x_anchor) in strips.items():
            for slot_idx, center in enumerate(slots):
                for level in range(1, self._rect_levels):
                    ratio = float(level) / float(self._rect_levels - 1)
                    width = self._rect_w_min + (self._rect_w_max - self._rect_w_min) * ratio
                    mask_img = np.zeros((tile_size, tile_size), dtype=np.uint8)
                    if y_anchor is not None:
                        xmin = center - self._rect_h
                        xmax = center + self._rect_h
                        if d_idx == MiniGridDirection.UP:
                            ymin = y_anchor - width
                            ymax = y_anchor
                        else:
                            ymin = y_anchor
                            ymax = y_anchor + width
                    else:
                        ymin = center - self._rect_h
                        ymax = center + self._rect_h
                        if d_idx == MiniGridDirection.LEFT:
                            xmin = x_anchor - width
                            xmax = x_anchor
                        else:
                            xmin = x_anchor
                            xmax = x_anchor + width
                    fill_coords(mask_img, point_in_rect(xmin, xmax, ymin, ymax), 1)
                    masks[d_idx, slot_idx, level] = mask_img.astype(bool)
        self._rect_mask_cache[tile_size] = masks
        return masks

    def _get_center_square_mask(self, tile_size: int) -> np.ndarray:
        mask = self._center_square_mask_cache.get(tile_size)
        if mask is not None:
            return mask
        mask_img = np.zeros((tile_size, tile_size), dtype=np.uint8)
        fill_coords(mask_img, point_in_rect(0.42, 0.58, 0.42, 0.58), 1)
        mask = mask_img.astype(bool)
        self._center_square_mask_cache[tile_size] = mask
        return mask


class _Ring12Renderer(BaseRenderer):
    _SECTOR_DA_ORDER: tuple[tuple[MiniGridDirection, MiniGridAction], ...] = (
        (MiniGridDirection.RIGHT, MiniGridAction.LEFT),
        (MiniGridDirection.RIGHT, MiniGridAction.FORWARD),
        (MiniGridDirection.RIGHT, MiniGridAction.RIGHT),
        (MiniGridDirection.DOWN, MiniGridAction.LEFT),
        (MiniGridDirection.DOWN, MiniGridAction.FORWARD),
        (MiniGridDirection.DOWN, MiniGridAction.RIGHT),
        (MiniGridDirection.LEFT, MiniGridAction.LEFT),
        (MiniGridDirection.LEFT, MiniGridAction.FORWARD),
        (MiniGridDirection.LEFT, MiniGridAction.RIGHT),
        (MiniGridDirection.UP, MiniGridAction.LEFT),
        (MiniGridDirection.UP, MiniGridAction.FORWARD),
        (MiniGridDirection.UP, MiniGridAction.RIGHT),
    )

    def __init__(
        self,
        alpha_min: float = 0.08,
        alpha_max: float = 0.85,
        ring_color: tuple[int, int, int] = (70, 190, 255),
        ring_inner: float = 0.24,
        ring_outer: float = 0.46,
    ) -> None:
        super().__init__()
        self._alpha_min = float(alpha_min)
        self._alpha_max = float(alpha_max)
        self._ring_color = np.asarray(ring_color, dtype=np.float32)
        self._ring_inner = float(ring_inner)
        self._ring_outer = float(ring_outer)
        self._sector_map_cache: dict[int, np.ndarray] = {}
        self._center_circle_mask_cache: dict[int, np.ndarray] = {}

    def render(
        self,
        tile_img_u8: np.ndarray,
        cell_values: np.ndarray,
        vmin: float,
        vmax: float,
    ) -> None:
        tile_size = int(tile_img_u8.shape[0])
        sector_map = self._get_sector_map(tile_size)
        norm_grid = self._normalize_values(
            self._flatten_directional(cell_values), vmin, vmax
        ).reshape(4, 3)
        alpha_values = np.zeros(12, dtype=np.float32)
        for sector_idx, (d_idx, a_idx) in enumerate(self._SECTOR_DA_ORDER):
            alpha_values[sector_idx] = (
                self._alpha_min + (self._alpha_max - self._alpha_min) * norm_grid[d_idx, a_idx]
            )
        tile_img = tile_img_u8.astype(np.float32)
        valid = sector_map >= 0
        if np.any(valid):
            alpha_map = np.zeros((tile_size, tile_size), dtype=np.float32)
            alpha_map[valid] = alpha_values[sector_map[valid]]
            tile_img = (1.0 - alpha_map[..., None]) * tile_img + alpha_map[..., None] * self._ring_color
        center_mask = self._get_center_circle_mask(tile_size)
        tile_img[center_mask] = self._pickup_scaled_color(cell_values, vmin, vmax)
        tile_img_u8[:, :, :] = np.clip(tile_img, 0.0, 255.0).astype(np.uint8)

    def _get_sector_map(self, tile_size: int) -> np.ndarray:
        sector_map = self._sector_map_cache.get(tile_size)
        if sector_map is not None:
            return sector_map
        yy, xx = np.indices((tile_size, tile_size), dtype=np.float32)
        cx = (tile_size - 1) * 0.5
        cy = (tile_size - 1) * 0.5
        dx = (xx - cx) / max(cx, 1.0)
        dy = (yy - cy) / max(cy, 1.0)
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(-dy, dx)
        clock = np.mod((math.pi / 2.0) - theta + 2.0 * math.pi, 2.0 * math.pi)
        step = 2.0 * math.pi / 12.0
        sector0_boundary = (math.pi / 6.0) - (step / 2.0)
        sector = np.floor(np.mod(clock - sector0_boundary, 2.0 * math.pi) / step).astype(np.int16)
        ring_mask = (r >= self._ring_inner) & (r <= self._ring_outer)
        sector_map = np.where(ring_mask, sector, -1).astype(np.int16)
        self._sector_map_cache[tile_size] = sector_map
        return sector_map

    def _get_center_circle_mask(self, tile_size: int) -> np.ndarray:
        mask = self._center_circle_mask_cache.get(tile_size)
        if mask is not None:
            return mask
        yy, xx = np.indices((tile_size, tile_size), dtype=np.float32)
        cx = (tile_size - 1) * 0.5
        cy = (tile_size - 1) * 0.5
        dx = (xx - cx) / max(cx, 1.0)
        dy = (yy - cy) / max(cy, 1.0)
        mask = (dx * dx + dy * dy) <= (0.13 * 0.13)
        self._center_circle_mask_cache[tile_size] = mask
        return mask


class DistributionWrapper(gym.Wrapper):
    """Render-time (s,a) distribution overlay for MiniGrid."""

    def __init__(
        self,
        env: gym.Env,
        distribution_layer: int = 20,
        overlay_name: str | None = None,
    ) -> None:
        overlay_wrapper = RenderOverlayWrapper.find_wrapper(env)
        if overlay_wrapper is None:
            raise TypeError("DistributionWrapper requires RenderOverlayWrapper in wrapper chain.")
        super().__init__(env)
        self._overlay_wrapper: RenderOverlayWrapper = overlay_wrapper
        self._overlay_name = overlay_name or f"distribution_{id(self)}"
        self._base_env = self.env.unwrapped
        self._obs_cache: dict[tuple[int, int, int], Any] = {}
        self._last_sa_values: np.ndarray | None = None
        self._q_function: Callable[[Any, int], float] | None = None
        self._frame_vmin = 0.0
        self._frame_vmax = 0.0
        self._render_style = "rect12"
        self._renderers: dict[str, BaseRenderer] = {
            "rect12": _Rect12Renderer(),
            "ring12": _Ring12Renderer(),
        }
        self._overlay_wrapper.register_overlay(
            self._overlay_name,
            int(distribution_layer),
            self._overlay_distribution,
        )

    def set_q_function(self, q_function: Callable[[Any, int], float] | None) -> None:
        self._q_function = q_function

    def register_renderer(self, style: str, renderer: BaseRenderer) -> None:
        self._renderers[str(style)] = renderer

    def set_render_style(self, style: str) -> None:
        if style not in self._renderers:
            choices = ", ".join(sorted(self._renderers))
            raise ValueError(f"style must be one of: {choices}")
        self._render_style = style

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._obs_cache.clear()
        self._last_sa_values = None
        self._frame_vmin = 0.0
        self._frame_vmax = 0.0
        return out

    def compute_sa_values(self, q_function: Callable[[Any, int], float]) -> np.ndarray:
        width = int(self._base_env.width)
        height = int(self._base_env.height)
        sa_values = np.zeros(
            (width, height, len(MiniGridDirection), len(MiniGridAction)),
            dtype=np.float32,
        )
        for x in range(width):
            for y in range(height):
                for d in MiniGridDirection:
                    obs_i = self._get_cached_observation(x, y, d)
                    for action in MiniGridAction:
                        sa_values[x, y, d, action] = float(q_function(obs_i, action))
        return sa_values

    def render(self):
        if self._q_function is not None:
            self._last_sa_values = self.compute_sa_values(self._q_function)
            self._frame_vmin = float(np.min(self._last_sa_values))
            self._frame_vmax = float(np.max(self._last_sa_values))
        else:
            self._last_sa_values = None
            self._frame_vmin = 0.0
            self._frame_vmax = 0.0
        return self.env.render()

    def _overlay_distribution(self, tile_img: np.ndarray, ctx: dict[str, Any]) -> None:
        if self._last_sa_values is None:
            return
        i = int(ctx["i"])
        j = int(ctx["j"])
        cell_values = self._last_sa_values[i, j]
        self._renderers[self._render_style].render(
            tile_img,
            cell_values,
            self._frame_vmin,
            self._frame_vmax,
        )

    def _get_cached_observation(self, x: int, y: int, d: int) -> Any:
        key = (int(x), int(y), int(d))
        obs = self._obs_cache.get(key)
        if obs is None:
            obs = minigrid_build_observation(key[0], key[1], key[2], self._base_env)
            self._obs_cache[key] = obs
        return obs
