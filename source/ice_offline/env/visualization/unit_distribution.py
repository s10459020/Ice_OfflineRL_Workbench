
import math
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.utils.rendering import fill_coords, point_in_rect

from ice_offline.dataset.old_value_collector import (
    MiniGridAction,
    MiniGridDirection,
    OldValueCollector,
)
from ice_offline.dataset.old_value_loader import OldValueLoader
from .overlay_engine import OverlayEngine, RenderLayer
from .overlay_loader import UnitLoaderInterface
from .overlay_renderer import UnitRenderer
from .overlay_wrapper import UnitWrapperInterface


# ====================
# Constants
# ====================
ACTION_ORDER_BY_DIRECTION: dict[MiniGridDirection, tuple[MiniGridAction, MiniGridAction, MiniGridAction]] = {
    MiniGridDirection.UP: (MiniGridAction.LEFT, MiniGridAction.FORWARD, MiniGridAction.RIGHT),
    MiniGridDirection.RIGHT: (MiniGridAction.LEFT, MiniGridAction.FORWARD, MiniGridAction.RIGHT),
    MiniGridDirection.DOWN: (MiniGridAction.RIGHT, MiniGridAction.FORWARD, MiniGridAction.LEFT),
    MiniGridDirection.LEFT: (MiniGridAction.RIGHT, MiniGridAction.FORWARD, MiniGridAction.LEFT),
}

STATE_ACTION_SECTOR_ORDER: tuple[tuple[MiniGridDirection, MiniGridAction], ...] = (
    (MiniGridDirection.UP, MiniGridAction.LEFT),
    (MiniGridDirection.UP, MiniGridAction.FORWARD),
    (MiniGridDirection.UP, MiniGridAction.RIGHT),
    (MiniGridDirection.RIGHT, MiniGridAction.LEFT),
    (MiniGridDirection.RIGHT, MiniGridAction.FORWARD),
    (MiniGridDirection.RIGHT, MiniGridAction.RIGHT),
    (MiniGridDirection.DOWN, MiniGridAction.LEFT),
    (MiniGridDirection.DOWN, MiniGridAction.FORWARD),
    (MiniGridDirection.DOWN, MiniGridAction.RIGHT),
    (MiniGridDirection.LEFT, MiniGridAction.LEFT),
    (MiniGridDirection.LEFT, MiniGridAction.FORWARD),
    (MiniGridDirection.LEFT, MiniGridAction.RIGHT),
)


# ====================
# Base Renderer
# ====================
class BasicStateActionRenderer(UnitRenderer):
    """State-action distribution base renderer.

    This base handles data ingestion and quantization.
    Subclasses only implement drawing style.
    """

    def __init__(
        self,
        *,
        quantize_mode: str = "percentile",
        value_min: float | None = None,
        value_max: float | None = None,
    ) -> None:
        super().__init__()
        assert quantize_mode in {"percentile", "fixed"}, "quantize_mode must be 'percentile' or 'fixed'"
        
        self._bins: np.ndarray | None = None
        self._edges: np.ndarray = np.empty((4,), dtype=np.float32)
        self._values: np.ndarray = np.empty((0,), dtype=np.float32)
        self._value_min = value_min
        self._value_max = value_max
        self._quantize_mode = quantize_mode
        self._quantize_levels = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float32)

    # ====================
    # Data Update
    # ====================
    def update(self, values: np.ndarray) -> None:
        self._values = values
        self._edges = self._compute_edges(self._values)
        self._bins = np.digitize(self._values, self._edges, right=True)

    # ====================
    # Data Access
    # ====================
    def in_map(self, i: int, j: int) -> bool:
        ix, iy = i - 1, j - 1
        return 0 <= ix < self._bins.shape[0] and 0 <= iy < self._bins.shape[1]

    def cell_bins(self, i: int, j: int) -> np.ndarray:
        return self._bins[i - 1, j - 1]

    # ====================
    # Quantize
    # ====================
    def _compute_edges(self, values: np.ndarray) -> np.ndarray:
        if self._quantize_mode == "fixed":
            v_min = np.min(values) if self._value_min is None else self._value_min
            v_max = np.max(values) if self._value_max is None else self._value_max
            return (v_min + (v_max - v_min) * self._quantize_levels).astype(np.float32)
        else:
            flat = values.reshape(-1)
            base = np.unique(flat)
            return np.quantile(base, self._quantize_levels).astype(np.float32)

    # ====================
    # UnitRenderer Hooks
    # ====================
    def condition_tile(self, *, i: int, j: int, tile_size: int) -> bool:
        return self.in_map(i, j)

    def cache_tile_key(self, *, i: int, j: int, tile_size: int) -> tuple[int, ...]:
        bins = self.cell_bins(i, j)
        return (tile_size, *tuple(bins.flatten()))


# ====================
# Ring Renderer
# ====================
class RingStateActionRenderer(BasicStateActionRenderer):
    """Ring style state-action renderer."""

    def __init__(
        self,
        *,
        quantize_mode: str = "percentile",
        value_min: float | None = None,
        value_max: float | None = None,
    ) -> None:
        super().__init__(quantize_mode=quantize_mode, value_min=value_min, value_max=value_max)
        self._ring_color = (70, 190, 255)
        self._pickup_color = (80, 220, 120)
        self._alpha_palette = (0.08, 0.2725, 0.465, 0.6575, 0.85)
        self._pickup_alpha_palette = (0.0, 0.25, 0.45, 0.65, 0.85)
        self._ring_sector_cache: dict[int, np.ndarray] = {}
        self._pickup_edge_mask_cache: dict[int, np.ndarray] = {}

    # ====================
    # UnitRenderer Hooks
    # ====================
    def overlay_tile(self, tile_img: np.ndarray, *, i: int, j: int, tile_size: int) -> None:
        bins = self.cell_bins(i, j)
        self._render_ring(tile_img, bins, tile_size)
        pickup_bins = bins[:, MiniGridAction.PICKUP]
        self._render_pickup(tile_img, pickup_bins, tile_size)

    # ====================
    # Style Render
    # ====================
    def _render_ring(self, tile_img: np.ndarray, bins: np.ndarray, tile_size: int) -> None:
        # 5-level bin -> alpha palette, then mapped into 12 ring sectors.
        sector_map = self._get_ring_sector_map(tile_size)
        alpha_values = np.zeros(12, dtype=np.float32)
        for sector_idx, (d_idx, a_idx) in enumerate(STATE_ACTION_SECTOR_ORDER):
            alpha_values[sector_idx] = self._alpha_palette[bins[d_idx, a_idx]]
        valid = sector_map >= 0
        if np.any(valid):
            alpha_map = np.zeros((tile_size, tile_size), dtype=np.float32)
            alpha_map[valid] = alpha_values[sector_map[valid]]
            tile = tile_img.astype(np.float32)
            tile_img[:, :, :] = np.clip(
                (1.0 - alpha_map[..., None]) * tile + alpha_map[..., None] * self._ring_color,
                0.0,
                255.0,
            ).astype(np.uint8)

    def _render_pickup(self, tile_img: np.ndarray, pickup_bins: np.ndarray, tile_size: int) -> None:
        masks = self._get_pickup_edge_masks(tile_size)
        for d_idx in MiniGridDirection:
            self._blend_mask(tile_img, masks[d_idx], self._pickup_color, self._pickup_alpha_palette[int(pickup_bins[d_idx])])

    def _get_ring_sector_map(self, tile_size: int) -> np.ndarray:
        sector_map = self._ring_sector_cache.get(tile_size)
        if sector_map is None:
            yy, xx = np.indices((tile_size, tile_size), dtype=np.float32)
            cx = (tile_size - 1) * 0.5
            cy = (tile_size - 1) * 0.5
            dx = (xx - cx) / max(cx, 1.0)
            dy = (yy - cy) / max(cy, 1.0)
            r = np.sqrt(dx * dx + dy * dy)
            theta = np.arctan2(-dy, dx)
            clock = np.mod((math.pi / 2.0) - theta + 2.0 * math.pi, 2.0 * math.pi)
            step = 2.0 * math.pi / 12.0
            sector0_boundary = -((step+1) / 2.0)
            sector = np.floor(np.mod(clock - sector0_boundary, 2.0 * math.pi) / step).astype(np.int16)
            ring_mask = (r >= 0.24) & (r <= 0.46)
            sector_map = np.where(ring_mask, sector, -1).astype(np.int16)
            self._ring_sector_cache[tile_size] = sector_map
        return sector_map

    def _get_pickup_edge_masks(self, tile_size: int) -> np.ndarray:
        masks = self._pickup_edge_mask_cache.get(tile_size)
        if masks is not None:
            return masks
        masks = np.zeros((4, tile_size, tile_size), dtype=bool)
        rects = {
            MiniGridDirection.UP: (0.42, 0.58, 0.08, 0.20),
            MiniGridDirection.RIGHT: (0.80, 0.92, 0.42, 0.58),
            MiniGridDirection.DOWN: (0.42, 0.58, 0.80, 0.92),
            MiniGridDirection.LEFT: (0.08, 0.20, 0.42, 0.58),
        }
        for d_idx, (xmin, xmax, ymin, ymax) in rects.items():
            mask_img = np.zeros((tile_size, tile_size), dtype=np.uint8)
            fill_coords(mask_img, point_in_rect(xmin, xmax, ymin, ymax), 1)
            masks[d_idx] = mask_img.astype(bool)
        self._pickup_edge_mask_cache[tile_size] = masks
        return masks

    def _blend_mask(self, tile_img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
        if alpha <= 0.0 or not np.any(mask):
            return
        tile = tile_img.astype(np.float32)
        color_arr = np.array(color, dtype=np.float32)
        tile[mask] = (1.0 - alpha) * tile[mask] + alpha * color_arr
        tile_img[:, :, :] = np.clip(tile, 0.0, 255.0).astype(np.uint8)


# ====================
# Rect Renderer
# ====================
class RectStateActionRenderer(BasicStateActionRenderer):
    """Rect style state-action renderer."""

    def __init__(
        self,
        *,
        quantize_mode: str = "percentile",
        value_min: float | None = None,
        value_max: float | None = None,
    ) -> None:
        super().__init__(quantize_mode=quantize_mode, value_min=value_min, value_max=value_max)
        self._rect_color = (255, 180, 60)
        self._pickup_color = (80, 220, 120)
        self._pickup_alpha_palette = (0.0, 0.25, 0.45, 0.65, 0.85)
        self._rect_mask_cache: dict[int, np.ndarray] = {}
        self._pickup_edge_mask_cache: dict[int, np.ndarray] = {}

    # ====================
    # UnitRenderer Hooks
    # ====================
    def overlay_tile(self, tile_img: np.ndarray, *, i: int, j: int, tile_size: int) -> None:
        bins = self.cell_bins(i, j)
        self._render_rect(tile_img, bins)
        pickup_bins = bins[:, MiniGridAction.PICKUP]
        self._render_pickup(tile_img, pickup_bins)

    # ====================
    # Style Render
    # ====================
    def _render_rect(self, tile_img: np.ndarray, bins: np.ndarray) -> None:
        # For each direction, draw 3 action strips with level-based width.
        masks = self._get_rect_masks(tile_img.shape[0])
        for d_idx in MiniGridDirection:
            action_order = ACTION_ORDER_BY_DIRECTION[d_idx]
            for slot_idx, a_real in enumerate(action_order):
                level = bins[d_idx, a_real]
                tile_img[masks[d_idx, slot_idx, level]] = self._rect_color

    def _render_pickup(self, tile_img: np.ndarray, pickup_bins: np.ndarray) -> None:
        masks = self._get_pickup_edge_masks(tile_img.shape[0])
        for d_idx in MiniGridDirection:
            self._blend_mask(tile_img, masks[d_idx], self._pickup_color, self._pickup_alpha_palette[int(pickup_bins[d_idx])])

    def _get_pickup_edge_masks(self, tile_size: int) -> np.ndarray:
        masks = self._pickup_edge_mask_cache.get(tile_size)
        if masks is not None:
            return masks
        masks = np.zeros((4, tile_size, tile_size), dtype=bool)
        rects = {
            MiniGridDirection.UP: (0.42, 0.58, 0.08, 0.20),
            MiniGridDirection.RIGHT: (0.80, 0.92, 0.42, 0.58),
            MiniGridDirection.DOWN: (0.42, 0.58, 0.80, 0.92),
            MiniGridDirection.LEFT: (0.08, 0.20, 0.42, 0.58),
        }
        for d_idx, (xmin, xmax, ymin, ymax) in rects.items():
            mask_img = np.zeros((tile_size, tile_size), dtype=np.uint8)
            fill_coords(mask_img, point_in_rect(xmin, xmax, ymin, ymax), 1)
            masks[d_idx] = mask_img.astype(bool)
        self._pickup_edge_mask_cache[tile_size] = masks
        return masks

    def _get_rect_masks(self, tile_size: int) -> np.ndarray:
        masks = self._rect_mask_cache.get(tile_size)
        if masks is not None:
            return masks
        levels = 5
        masks = np.zeros((4, 3, levels, tile_size, tile_size), dtype=bool)
        strips: dict[MiniGridDirection, tuple[list[float], float | None, float | None]] = {
            MiniGridDirection.UP: ([0.40, 0.50, 0.60], 0.30, None),
            MiniGridDirection.DOWN: ([0.40, 0.50, 0.60], 0.70, None),
            MiniGridDirection.LEFT: ([0.40, 0.50, 0.60], None, 0.30),
            MiniGridDirection.RIGHT: ([0.40, 0.50, 0.60], None, 0.70),
        }
        for d_idx, (slots, y_anchor, x_anchor) in strips.items():
            for slot_idx, center in enumerate(slots):
                for level in range(levels):
                    ratio = level / 4.0
                    width = 0.015 + (0.135 - 0.015) * ratio
                    mask_img = np.zeros((tile_size, tile_size), dtype=np.uint8)
                    if y_anchor is not None:
                        xmin = center - 0.045
                        xmax = center + 0.045
                        ymin, ymax = (y_anchor - width, y_anchor) if d_idx == MiniGridDirection.UP else (y_anchor, y_anchor + width)
                    else:
                        ymin = center - 0.045
                        ymax = center + 0.045
                        xmin, xmax = (x_anchor - width, x_anchor) if d_idx == MiniGridDirection.LEFT else (x_anchor, x_anchor + width)
                    fill_coords(mask_img, point_in_rect(xmin, xmax, ymin, ymax), 1)
                    masks[d_idx, slot_idx, level] = mask_img.astype(bool)
        self._rect_mask_cache[tile_size] = masks
        return masks

    def _blend_mask(self, tile_img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
        if alpha <= 0.0 or not np.any(mask):
            return
        tile = tile_img.astype(np.float32)
        color_arr = np.array(color, dtype=np.float32)
        tile[mask] = (1.0 - alpha) * tile[mask] + alpha * color_arr
        tile_img[:, :, :] = np.clip(tile, 0.0, 255.0).astype(np.uint8)


# ====================
# Unit
# ====================
class DistributionUnit(UnitWrapperInterface, UnitLoaderInterface):
    """Collect values and register state-action renderer(s).

    Responsibilities:
    - ensure OldValueCollector exists on wrapper path
    - ingest info["values"] during on_render
    - register renderer(s) into OverlayEngine
    """

    def __init__(
        self,
        *,
        value_fn: Callable[[Any, int], float] | None = None,
        style: str = "ring",
        quantize_mode: str = "percentile",
        value_min: float | None = None,
        value_max: float | None = None,
    ) -> None:
        self._value_fn = value_fn
        # Unit chooses style renderer and acts as bridge between
        # runtime data source (wrapper/loader) and renderer update.
        if style == "ring":
            self._sa_renderer = RingStateActionRenderer(
                quantize_mode=quantize_mode,
                value_min=value_min,
                value_max=value_max,
            )
        elif style == "rect":
            self._sa_renderer = RectStateActionRenderer(
                quantize_mode=quantize_mode,
                value_min=value_min,
                value_max=value_max,
            )
        else:
            raise ValueError("style must be 'ring' or 'rect'")

    # ====================
    # Wrapper Hooks
    # ====================
    def on_wrapper(self, env: gym.Env, wrapper: Any, engine: OverlayEngine) -> gym.Env:
        # Online mode: register renderer and ensure values source exists.
        engine.register(int(RenderLayer.DISTRIBUTION), self._sa_renderer)
        collector = OldValueCollector(env, self._value_fn)
        wrapper.register("values", collector.get_last)
        return collector

    # ====================
    # Loader Hooks
    # ====================
    def on_loader(self, engine: OverlayEngine, loader: Any) -> None:
        # Offline mode: register renderer and load per-episode values list.
        engine.register(int(RenderLayer.DISTRIBUTION), self._sa_renderer)
        value_loader = OldValueLoader(loader.dataset_id)
        loader.register_list("values", lambda episode_index: value_loader.load_episode(episode_index))

    # ====================
    # Shared Hooks
    # ====================
    def on_render(self, data: dict[str, Any]) -> None:
        self._sa_renderer.update(data["values"])

