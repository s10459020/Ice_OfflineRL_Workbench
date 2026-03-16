import math
from collections.abc import Callable
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.utils.rendering import fill_coords, point_in_rect

from .render_overlay_wrapper import OverlayDependentWrapper

class MiniGridDirection(IntEnum):
    """MiniGrid direction order used by environment internals."""
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


class MiniGridAction(IntEnum):
    """Subset of actions visualized by this module."""
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    PICKUP = 3


class BaseRenderer:
    """Base utilities shared by concrete distribution renderers."""
    def __init__(
        self,
        gamma: float = 0.60,
        pickup_color: tuple[int, int, int] = (80, 220, 120),
        pickup_fill_min: float = 0.10,
    ) -> None:
        self._gamma = gamma
        self._pickup_color = np.asarray(pickup_color, dtype=np.float32)
        self._pickup_fill_min = pickup_fill_min

    @staticmethod
    def _flatten_directional(cell_values: np.ndarray) -> np.ndarray:
        """Flatten the 4x3 directional action values into 12 sectors."""
        return cell_values[:, :3].reshape(12)

    def _normalize_values(self, values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        """Normalize an array to [0, 1] with gamma shaping."""
        den = vmax - vmin
        if den <= 1e-12:
            return np.zeros_like(values, dtype=np.float32)
        return np.power((values - vmin) / den, self._gamma, dtype=np.float32)

    def _normalize_scalar(self, value: float, vmin: float, vmax: float) -> float:
        """Scalar version of `_normalize_values`."""
        den = vmax - vmin
        if den <= 1e-12:
            return 0.0
        norm = np.clip((value - vmin) / den, 0.0, 1.0)
        return np.power(norm, self._gamma)

    def _pickup_scaled_color(self, cell_values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        """Map pickup value strength to center marker color."""
        pickup_value = np.mean(cell_values[:, MiniGridAction.PICKUP])
        pickup_norm = self._normalize_scalar(pickup_value, vmin, vmax)
        pickup_scale = self._pickup_fill_min + (1.0 - self._pickup_fill_min) * pickup_norm
        return np.clip(self._pickup_color * pickup_scale, 0.0, 255.0)

    def render(
        self,
        tile_img_u8: np.ndarray,
        cell_values: np.ndarray,
        vmin: float,
        vmax: float,
    ) -> None:
        """Render one cell overlay into `tile_img_u8`."""
        pass


class _Rect12Renderer(BaseRenderer):
    """Render directional values as 12 bars around tile center."""
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
        self._rect_w_min = rect_w_min
        self._rect_w_max = rect_w_max
        self._rect_h = rect_h
        self._rect_levels = rect_levels
        self._rect_mask_cache: dict[int, np.ndarray] = {}
        self._center_square_mask_cache: dict[int, np.ndarray] = {}

    def render(
        self,
        tile_img_u8: np.ndarray,
        cell_values: np.ndarray,
        vmin: float,
        vmax: float,
    ) -> None:
        # Build 4x3 normalized values -> map each slot to bar length level.
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
                level = int(round(norm_grid[d_idx, a_real] * (self._rect_levels - 1)))
                level = max(1, min(self._rect_levels - 1, level))
                tile_img_u8[masks[d_idx, slot_idx, level]] = self._rect_color
        center_mask = self._get_center_square_mask(tile_img_u8.shape[0])
        # Center square visualizes pickup intensity.
        tile_img_u8[center_mask] = self._pickup_scaled_color(cell_values, vmin, vmax).astype(np.uint8)

    def _get_rect_masks(self, tile_size: int) -> np.ndarray:
        """Build/cache bar masks for each direction-slot-level combination."""
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
                    ratio = level / (self._rect_levels - 1)
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
        """Build/cache center square mask."""
        mask = self._center_square_mask_cache.get(tile_size)
        if mask is not None:
            return mask
        mask_img = np.zeros((tile_size, tile_size), dtype=np.uint8)
        fill_coords(mask_img, point_in_rect(0.42, 0.58, 0.42, 0.58), 1)
        mask = mask_img.astype(bool)
        self._center_square_mask_cache[tile_size] = mask
        return mask


class _Ring12Renderer(BaseRenderer):
    """Render directional values as alpha-weighted 12-sector ring."""
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
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._ring_color = np.asarray(ring_color, dtype=np.float32)
        self._ring_inner = ring_inner
        self._ring_outer = ring_outer
        self._sector_map_cache: dict[int, np.ndarray] = {}
        self._center_circle_mask_cache: dict[int, np.ndarray] = {}

    def render(
        self,
        tile_img_u8: np.ndarray,
        cell_values: np.ndarray,
        vmin: float,
        vmax: float,
    ) -> None:
        # Convert directional values to per-sector alpha, then blend ring color.
        tile_size = tile_img_u8.shape[0]
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
        # Center circle visualizes pickup intensity.
        tile_img[center_mask] = self._pickup_scaled_color(cell_values, vmin, vmax)
        tile_img_u8[:, :, :] = np.clip(tile_img, 0.0, 255.0).astype(np.uint8)

    def _get_sector_map(self, tile_size: int) -> np.ndarray:
        """Build/cache sector index map for ring pixels (-1 outside ring)."""
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
        """Build/cache center circle mask."""
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


class _StateVRenderer(BaseRenderer):
    """Render per-cell V(s) as a tile-wide alpha tint."""

    def __init__(
        self,
        state_color: tuple[int, int, int] = (255, 140, 80),
        alpha_min: float = 0.05,
        alpha_max: float = 0.90,
    ) -> None:
        super().__init__()
        self._state_color = np.asarray(state_color, dtype=np.float32)
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max

    def render(
        self,
        tile_img_u8: np.ndarray,
        cell_values: np.ndarray,
        vmin: float,
        vmax: float,
    ) -> None:
        # V(s): expected Q(s,a) over actions, then averaged over directions.
        # Here we use a uniform expectation over all four actions.
        v_s = float(np.mean(np.mean(cell_values[:, :4], axis=1)))
        v_norm = self._normalize_scalar(v_s, vmin, vmax)
        alpha = self._alpha_min + (self._alpha_max - self._alpha_min) * v_norm

        tile_img = tile_img_u8.astype(np.float32)
        tile_img = (1.0 - alpha) * tile_img + alpha * self._state_color
        tile_img_u8[:, :, :] = np.clip(tile_img, 0.0, 255.0).astype(np.uint8)


class DistributionWrapper(OverlayDependentWrapper):
    """
    Render-time value distribution overlay for MiniGrid.

    Wrapper responsibilities:
    1) Build per-state value map from a user-provided value function.
    2) Register one overlay callback via OverlayDependentWrapper.
    3) Delegate actual drawing to a fixed renderer chosen at initialization.
    """

    _RENDERER_TYPES: dict[str, type[BaseRenderer]] = {
        "rect12": _Rect12Renderer,
        "ring12": _Ring12Renderer,
        "state_v": _StateVRenderer,
    }

    def __init__(
        self,
        env: gym.Env,
        value_fn: Callable[[Any, int], float],
        layer: int = 20,
        style: str = "rect12",
    ) -> None:
        if style not in self._RENDERER_TYPES:
            choices = ", ".join(sorted(self._RENDERER_TYPES))
            raise ValueError(f"style must be one of: {choices}")

        super().__init__(
            env,
            overlay_fn=self._overlay_callback,
            overlay_layer=layer,
        )

        self._base_env = self.env.unwrapped
        # Observation cache keyed by (x, y, dir) for full-grid value sweeps.
        self._obs_cache: dict[tuple[int, int, int], Any] = {}
        self._wrappers_cache: list[gym.Wrapper] | None = None
        self._wrappers_cache_root: gym.Env | None = None
        # Latest frame value map (width, height, direction, action).
        self._values: np.ndarray | None = None
        self._values_min = 0.0
        self._values_max = 0.0
        # Value estimator used to fill the value map each frame.
        self._value_fn = value_fn
        # Rendering style is fixed at initialization time.
        self._renderer = self._RENDERER_TYPES[style]()

    # ---------------------------------------------------------------------
    # Env lifecycle
    # ---------------------------------------------------------------------
    def reset(self, **kwargs):
        """Clear frame and observation caches on environment reset."""
        out = self.env.reset(**kwargs)
        self._obs_cache.clear()
        self._wrappers_cache = None
        self._wrappers_cache_root = None
        self._values = None
        self._values_min = 0.0
        self._values_max = 0.0
        return out

    def render(self):
        """Refresh value map for current frame, then render downstream env."""
        # Cache frame-wise range so all cells share the same normalization scale.
        self._values = self._compute_distribution_values()
        self._values_min = np.min(self._values)
        self._values_max = np.max(self._values)
        return self.env.render()

    # ---------------------------------------------------------------------
    # Value map build
    # ---------------------------------------------------------------------
    def _compute_distribution_values(self) -> np.ndarray:
        """Compute full-grid value map for all directions/actions."""
        width = self._base_env.width
        height = self._base_env.height
        values = np.zeros(
            (width, height, len(MiniGridDirection), len(MiniGridAction)),
            dtype=np.float32,
        )
        for x in range(width):
            for y in range(height):
                for d in MiniGridDirection:
                    obs_i = self._get_cached_observation(x, y, d)
                    for action in MiniGridAction:
                        values[x, y, d, action] = self._value_fn(obs_i, action)
        return values

    def _get_cached_observation(self, x: int, y: int, d: int) -> Any:
        """Get cached observation for (x, y, d), building it on first access."""
        key = (x, y, d)
        obs_i = self._obs_cache.get(key)
        if obs_i is None:
            obs_i = self._build_observation(key[0], key[1], key[2])
            self._obs_cache[key] = obs_i
        return obs_i

    def _build_observation(self, x: int, y: int, d: int) -> Any:
        """
        Build an observation for an arbitrary (x, y, d) without stepping the env.

        The function temporarily rewrites agent position/direction, calls `gen_obs`,
        and restores the original state before returning.
        """
        old_pos = tuple(self._base_env.agent_pos)
        old_dir = self._base_env.agent_dir
        try:
            self._base_env.agent_pos = (x, y)
            self._base_env.agent_dir = d
            obs = self._base_env.gen_obs()
            # Apply observation transformations while forced (x, y, d) is active.
            for wrapper in reversed(self._get_cached_wrappers()):
                if isinstance(wrapper, gym.ObservationWrapper):
                    obs = wrapper.observation(obs)
        finally:
            self._base_env.agent_pos = old_pos
            self._base_env.agent_dir = old_dir
        return obs

    def _get_cached_wrappers(self) -> list[gym.Wrapper]:
        """
        Cache wrapper chain traversal from current outer env to base env.
        Rebuild cache when outer env object changes.
        """
        if self._wrappers_cache is not None and self._wrappers_cache_root is self.env:
            return self._wrappers_cache

        wrappers: list[gym.Wrapper] = []
        current: gym.Env = self.env
        while isinstance(current, gym.Wrapper):
            wrappers.append(current)
            current = current.env

        self._wrappers_cache = wrappers
        self._wrappers_cache_root = self.env
        return wrappers

    # ---------------------------------------------------------------------
    # Overlay callback
    # ---------------------------------------------------------------------
    def _overlay_callback(self, tile_img: np.ndarray, ctx: dict[str, Any]) -> None:
        """Overlay callback invoked by RenderOverlayWrapper per tile."""
        if self._values is None:
            return
        i = ctx["i"]
        j = ctx["j"]
        cell_values = self._values[i, j]
        self._renderer.render(
            tile_img,
            cell_values,
            self._values_min,
            self._values_max,
        )
