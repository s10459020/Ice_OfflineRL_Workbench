from __future__ import annotations

import math
from collections.abc import Callable
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.utils.rendering import fill_coords, point_in_rect

from ..model.state import State
from .overlay_engine import OverlayEngine, RenderLayer, UnitRegisterInterface
from .overlay_loader import UnitLoaderInterface
from .overlay_renderer import UnitRenderer
from .overlay_wrapper import UnitWrapperInterface


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# Renderer Base
# ------------------------------------------------------------------
class BaseRenderer:
    def __init__(
        self,
        n_segments: int = 5,
        pickup_color: tuple[int, int, int] = (80, 220, 120),
    ) -> None:
        if n_segments != 5:
            raise ValueError("n_segments is fixed to 5 (quantile bins: 20/40/60/80).")
        self._n_segments = n_segments
        self._quantile_levels = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
        self._pickup_color = np.asarray(pickup_color, dtype=np.float32)
        self._pickup_mask_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _flatten_directional(cell_values: np.ndarray) -> np.ndarray:
        return cell_values[:, :3].reshape(12)

    def compute_quantile_edges(self, frame_values: np.ndarray) -> np.ndarray:
        flat = np.asarray(frame_values, dtype=np.float32).reshape(-1)
        edges = np.quantile(flat, self._quantile_levels).astype(np.float32)
        if float(edges[-1] - edges[0]) <= 1e-12:
            v_min = float(np.min(flat))
            v_max = float(np.max(flat))
            if float(v_max - v_min) > 1e-12:
                edges = np.linspace(v_min, v_max, 6, dtype=np.float32)[1:-1]
        return edges

    @staticmethod
    def quantize_values(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        edges = np.asarray(edges, dtype=np.float32).reshape(-1)
        if edges.size != 4:
            raise ValueError("edges must contain exactly 4 quantile thresholds.")
        bins = np.digitize(values, edges, right=True)
        return np.clip(bins, 0, 4).astype(np.int8)

    @classmethod
    def quantize_scalar(cls, value: float, edges: np.ndarray) -> int:
        return int(cls.quantize_values(np.asarray([value], dtype=np.float32), edges)[0])

    def render(self, tile_img_u8: np.ndarray, cell_values: np.ndarray, quantile_edges: np.ndarray) -> None:
        raise NotImplementedError

    @staticmethod
    def _blend_mask(tile_img_u8: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
        if alpha <= 0.0 or not np.any(mask):
            return
        tile_f = tile_img_u8.astype(np.float32)
        tile_f[mask] = (1.0 - alpha) * tile_f[mask] + alpha * color
        tile_img_u8[:, :, :] = np.clip(tile_f, 0.0, 255.0).astype(np.uint8)

    def _get_pickup_masks(self, tile_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        masks = self._pickup_mask_cache.get(tile_size)
        if masks is not None:
            return masks

        yy, xx = np.indices((tile_size, tile_size), dtype=np.float32)
        cx = (tile_size - 1) * 0.5
        cy = (tile_size - 1) * 0.5
        dx = (xx - cx) / max(cx, 1.0)
        dy = (yy - cy) / max(cy, 1.0)

        circle_mask = (dx * dx + dy * dy) <= (0.16 * 0.16)

        square_img = np.zeros((tile_size, tile_size), dtype=np.uint8)
        fill_coords(square_img, point_in_rect(0.42, 0.58, 0.42, 0.58), 1)
        square_mask = square_img.astype(bool)

        square_max_img = np.zeros((tile_size, tile_size), dtype=np.uint8)
        fill_coords(square_max_img, point_in_rect(0.36, 0.64, 0.36, 0.64), 1)
        square_max_mask = square_max_img.astype(bool)

        masks = (circle_mask, square_mask, square_max_mask)
        self._pickup_mask_cache[tile_size] = masks
        return masks

    def _render_pickup_marker(
        self,
        tile_img_u8: np.ndarray,
        cell_values: np.ndarray,
        quantile_edges: np.ndarray,
    ) -> None:
        pickup_value = float(np.mean(cell_values[:, MiniGridAction.PICKUP]))
        pickup_bin = self.quantize_scalar(pickup_value, quantile_edges)
        if pickup_bin <= 0:
            return

        circle_mask, square_mask, square_max_mask = self._get_pickup_masks(tile_img_u8.shape[0])
        if pickup_bin == 1:
            self._blend_mask(tile_img_u8, circle_mask, self._pickup_color, alpha=0.45)
        elif pickup_bin == 2:
            self._blend_mask(tile_img_u8, circle_mask, self._pickup_color, alpha=0.90)
        elif pickup_bin == 3:
            self._blend_mask(tile_img_u8, square_mask, self._pickup_color, alpha=0.90)
        else:
            self._blend_mask(tile_img_u8, square_max_mask, self._pickup_color, alpha=0.95)


# ------------------------------------------------------------------
# Renderer: Rect12
# ------------------------------------------------------------------
class _Rect12Renderer(BaseRenderer):
    def __init__(
        self,
        rect_color: tuple[int, int, int] = (255, 180, 60),
        rect_w_min: float = 0.015,
        rect_w_max: float = 0.135,
        rect_h: float = 0.045,
    ) -> None:
        super().__init__()
        self._rect_color = np.asarray(rect_color, dtype=np.uint8)
        self._rect_w_min = rect_w_min
        self._rect_w_max = rect_w_max
        self._rect_h = rect_h
        self._rect_levels = self._n_segments
        self._rect_mask_cache: dict[int, np.ndarray] = {}

    def render(self, tile_img_u8: np.ndarray, cell_values: np.ndarray, quantile_edges: np.ndarray) -> None:
        bin_grid = self.quantize_values(self._flatten_directional(cell_values), quantile_edges).reshape(4, 3)
        masks = self._get_rect_masks(tile_img_u8.shape[0])
        action_order = (MiniGridAction.LEFT, MiniGridAction.FORWARD, MiniGridAction.RIGHT)
        for d_idx in MiniGridDirection:
            for slot_idx, a_real in enumerate(action_order):
                level = int(bin_grid[d_idx, a_real])
                tile_img_u8[masks[d_idx, slot_idx, level]] = self._rect_color
        self._render_pickup_marker(tile_img_u8, cell_values, quantile_edges)

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
                for level in range(self._rect_levels):
                    ratio = level / max(1, (self._rect_levels - 1))
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


# ------------------------------------------------------------------
# Renderer: Ring12
# ------------------------------------------------------------------
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
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._ring_color = np.asarray(ring_color, dtype=np.float32)
        self._ring_inner = ring_inner
        self._ring_outer = ring_outer
        self._sector_map_cache: dict[int, np.ndarray] = {}

    def render(self, tile_img_u8: np.ndarray, cell_values: np.ndarray, quantile_edges: np.ndarray) -> None:
        tile_size = tile_img_u8.shape[0]
        sector_map = self._get_sector_map(tile_size)
        bin_grid = self.quantize_values(self._flatten_directional(cell_values), quantile_edges).reshape(4, 3)
        alpha_palette = np.asarray(
            [self._alpha_min + (self._alpha_max - self._alpha_min) * (k / 4.0) for k in range(5)],
            dtype=np.float32,
        )
        alpha_values = np.zeros(12, dtype=np.float32)
        for sector_idx, (d_idx, a_idx) in enumerate(self._SECTOR_DA_ORDER):
            alpha_values[sector_idx] = alpha_palette[int(bin_grid[d_idx, a_idx])]
        tile_img = tile_img_u8.astype(np.float32)
        valid = sector_map >= 0
        if np.any(valid):
            alpha_map = np.zeros((tile_size, tile_size), dtype=np.float32)
            alpha_map[valid] = alpha_values[sector_map[valid]]
            tile_img = (1.0 - alpha_map[..., None]) * tile_img + alpha_map[..., None] * self._ring_color
        tile_img_u8[:, :, :] = np.clip(tile_img, 0.0, 255.0).astype(np.uint8)
        self._render_pickup_marker(tile_img_u8, cell_values, quantile_edges)

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


# ------------------------------------------------------------------
# Renderer: State-V
# ------------------------------------------------------------------
class _StateVRenderer(BaseRenderer):
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

    def render(self, tile_img_u8: np.ndarray, cell_values: np.ndarray, quantile_edges: np.ndarray) -> None:
        v_s = float(np.mean(np.mean(cell_values[:, :4], axis=1)))
        v_bin = self.quantize_scalar(v_s, quantile_edges)
        alpha = self._alpha_min + (self._alpha_max - self._alpha_min) * (v_bin / 4.0)
        tile_img = tile_img_u8.astype(np.float32)
        tile_img = (1.0 - alpha) * tile_img + alpha * self._state_color
        tile_img_u8[:, :, :] = np.clip(tile_img, 0.0, 255.0).astype(np.uint8)
        self._render_pickup_marker(tile_img_u8, cell_values, quantile_edges)


# ------------------------------------------------------------------
# Distribution Unit
# ------------------------------------------------------------------
class DistributionUnit(UnitWrapperInterface, UnitLoaderInterface, UnitRegisterInterface, UnitRenderer):
    _CANDIDATE_KEYS = ("q_values", "q_table", "qtable", "distribution")
    _RENDERER_TYPES: dict[str, type[BaseRenderer]] = {
        "rect12": _Rect12Renderer,
        "ring12": _Ring12Renderer,
        "state_v": _StateVRenderer,
    }

    def __init__(
        self,
        *,
        value_fn: Callable[[Any, int], float] | None = None,
        style: str = "rect12",
    ) -> None:
        UnitRenderer.__init__(self)
        if style not in self._RENDERER_TYPES:
            choices = ", ".join(sorted(self._RENDERER_TYPES))
            raise ValueError(f"style must be one of: {choices}")
        self._value_fn = value_fn
        self._renderer = self._RENDERER_TYPES[style]()

        # Wrapper mode (online): build values via value_fn and env observation synthesis.
        self._base_env: gym.Env | None = None
        self._obs_cache: dict[tuple[int, int, int], Any] = {}

        # Loader mode (offline): values are read from infos payload.
        self._infos: list[dict[str, Any]] = []

        # Shared render state
        self._values: np.ndarray | None = None  # (inner_w, inner_h, 4, 4)
        self._quantile_edges = np.zeros(4, dtype=np.float32)

    # ------------------------------------------------------------------
    # Engine Registration
    # ------------------------------------------------------------------
    def register_engine(self, engine: OverlayEngine) -> None:
        engine.register(int(RenderLayer.QTABLE), self)

    # ------------------------------------------------------------------
    # Wrapper/Loader Hooks
    # ------------------------------------------------------------------
    def on_env(self, env: gym.Env) -> None:
        self._base_env = env

    def on_reset(self, state: State, info: dict[str, Any]) -> None:
        del state, info
        self._obs_cache.clear()
        self._values = None
        self._quantile_edges.fill(0.0)

    def on_step(self, state: State, action: Any, reward: float, done: bool, info: dict[str, Any]) -> None:
        del state, action, reward, done, info

    def on_render(self, state: State, info: dict[str, Any]) -> None:
        del state, info
        if self._value_fn is None:
            return
        self._values = self._compute_distribution_values()
        if self._values is None or self._values.size == 0:
            self._quantile_edges.fill(0.0)
            return
        shown_values = self._values[:, :, :, :4]
        self._quantile_edges = self._renderer.compute_quantile_edges(shown_values)

    def on_load(
        self,
        states: list[State],
        actions: list[Any],
        rewards: list[float],
        dones: list[bool],
        infos: list[dict[str, Any]],
    ) -> None:
        del states, actions, rewards, dones
        self._infos = infos
        if self._value_fn is None:
            self._update_values_from_infos(0)

    def on_seek(self, transition_index: int) -> None:
        if self._value_fn is None:
            self._update_values_from_infos(int(transition_index))

    # ------------------------------------------------------------------
    # Render Hooks
    # ------------------------------------------------------------------
    def condition_tile(self, *, i: int, j: int, tile_size: int) -> bool:
        del tile_size
        if self._values is None:
            return False
        ix, iy = int(i) - 1, int(j) - 1
        return 0 <= ix < self._values.shape[0] and 0 <= iy < self._values.shape[1]

    def cache_tile_key(self, *, i: int, j: int, tile_size: int) -> tuple[int, ...] | None:
        del tile_size
        if not self.condition_tile(i=i, j=j, tile_size=0):
            return None
        cell_values = self._values[int(i) - 1, int(j) - 1]
        bins = BaseRenderer.quantize_values(
            np.asarray(cell_values[:, :4], dtype=np.float32).reshape(-1),
            self._quantile_edges,
        )
        return tuple(int(v) for v in bins)

    def overlay_tile(self, tile_img: np.ndarray, *, i: int, j: int, tile_size: int) -> None:
        del tile_size
        if not self.condition_tile(i=i, j=j, tile_size=0):
            return
        cell_values = np.asarray(self._values[int(i) - 1, int(j) - 1], dtype=np.float32)
        self._renderer.render(tile_img, cell_values, self._quantile_edges)

    # ------------------------------------------------------------------
    # Wrapper Mode Helpers
    # ------------------------------------------------------------------
    def _compute_distribution_values(self) -> np.ndarray | None:
        if self._base_env is None or self._value_fn is None:
            return None
        width = int(self._base_env.width)
        height = int(self._base_env.height)
        inner_w = max(0, width - 2)
        inner_h = max(0, height - 2)
        values = np.zeros(
            (inner_w, inner_h, len(MiniGridDirection), len(MiniGridAction)),
            dtype=np.float32,
        )
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                ix = x - 1
                iy = y - 1
                for d in MiniGridDirection:
                    obs_i = self._get_cached_observation(x, y, d)
                    for action in MiniGridAction:
                        values[ix, iy, d, action] = self._value_fn(obs_i, int(action))
        return values

    def _get_cached_observation(self, x: int, y: int, d: int) -> Any:
        key = (x, y, d)
        obs_i = self._obs_cache.get(key)
        if obs_i is None:
            obs_i = self._build_observation(x, y, d)
            self._obs_cache[key] = obs_i
        return obs_i

    def _build_observation(self, x: int, y: int, d: int) -> Any:
        if self._base_env is None:
            raise RuntimeError("base env is not initialized")
        old_pos = tuple(self._base_env.agent_pos)
        old_dir = int(self._base_env.agent_dir)
        try:
            self._base_env.agent_pos = (x, y)
            self._base_env.agent_dir = d
            return self._base_env.gen_obs()
        finally:
            self._base_env.agent_pos = old_pos
            self._base_env.agent_dir = old_dir

    # ------------------------------------------------------------------
    # Loader Mode Helpers
    # ------------------------------------------------------------------
    def _update_values_from_infos(self, step: int) -> None:
        self._values = None
        self._quantile_edges.fill(0.0)
        if not self._infos or step < 0 or step >= len(self._infos):
            return
        raw = self._find_candidate(self._infos[step])
        if raw is None:
            return
        values = self._coerce_values(raw)
        if values is None or values.size == 0:
            return
        self._values = values
        shown_values = values[:, :, :, :4]
        self._quantile_edges = self._renderer.compute_quantile_edges(shown_values)

    def _coerce_values(self, raw: Any) -> np.ndarray | None:
        values = np.asarray(raw, dtype=np.float32)
        if values.ndim != 4:
            return None
        if values.shape[2] != 4:
            return None
        if values.shape[3] < 4:
            return None
        if values.shape[3] > 4:
            values = values[:, :, :, :4]
        return values

    def _find_candidate(self, payload: Any) -> Any | None:
        if isinstance(payload, dict):
            for key in self._CANDIDATE_KEYS:
                if key in payload:
                    return payload[key]
            for value in payload.values():
                found = self._find_candidate(value)
                if found is not None:
                    return found
        return None
