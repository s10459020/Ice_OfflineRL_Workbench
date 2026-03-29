from enum import IntEnum
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
import time
import numpy as np


# ------------------------------------------------------------------
# Render Layers
# ------------------------------------------------------------------
class RenderLayer(IntEnum):
    BACKGROUND = 0
    OBJECTS = 10
    TRAIL = 30
    AGENT = 40
    HIGHLIGHT = 50


class UnitRenderer(ABC):
    def __init__(
        self,
        *,
        cache_one_tile: bool = False,
        cache_one_frame: bool = False,
    ) -> None:
        self._cache_one_tile = bool(cache_one_tile)
        self._cache_one_frame = bool(cache_one_frame)
        self._tile_cache: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
        self._frame_cache: dict[Any, np.ndarray] = {}
        self._one_tile_key: Any = None
        self._one_tile_img: np.ndarray | None = None
        self._one_tile_mask: np.ndarray | None = None
        self._one_frame_key: Any = None
        self._one_frame_img: np.ndarray | None = None

    # ------------------------------------------------------------------
    # cache
    # ------------------------------------------------------------------
    def condition_tile(self, *, i: int, j: int, tile_size: int) -> bool:
        return True

    def condition_frame(self, *, grid_width: int, grid_height: int, tile_size: int) -> bool:
        return True

    def cache_tile_key(self, *, i: int, j: int, tile_size: int) -> Any | None:
        return None

    def cache_frame_key(self, *, grid_width: int, grid_height: int, tile_size: int) -> Any | None:
        return None

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------
    def render_tile(self, tile_img: np.ndarray, *, i: int, j: int, tile_size: int) -> None:
        if not self.condition_tile(i=i, j=j, tile_size=tile_size):
            return

        key = self.cache_tile_key(i=i, j=j, tile_size=tile_size)
        if key is None:
            self.overlay_tile(tile_img, i=i, j=j, tile_size=tile_size)
            return

        if self._cache_one_tile:
            if self._one_tile_key == key:
                tile_img[self._one_tile_mask] = self._one_tile_img[self._one_tile_mask]
                return
            drawn = np.zeros_like(tile_img)
            self.overlay_tile(drawn, i=i, j=j, tile_size=tile_size)
            mask = np.any(drawn != 0, axis=2)
            self._one_tile_key = key
            self._one_tile_img = drawn
            self._one_tile_mask = mask
            tile_img[mask] = drawn[mask]
            return

        cached = self._tile_cache.get(key)
        if cached is not None:
            drawn, mask = cached
            tile_img[mask] = drawn[mask]
            return

        drawn = np.zeros_like(tile_img)
        self.overlay_tile(drawn, i=i, j=j, tile_size=tile_size)
        mask = np.any(drawn != 0, axis=2)
        self._tile_cache[key] = (drawn, mask)
        tile_img[mask] = drawn[mask]

    def render_frame(
        self,
        frame_img: np.ndarray,
        *,
        grid_width: int,
        grid_height: int,
        tile_size: int,
    ) -> None:
        if not self.condition_frame(grid_width=grid_width, grid_height=grid_height, tile_size=tile_size):
            return

        key = self.cache_frame_key(grid_width=grid_width, grid_height=grid_height, tile_size=tile_size)
        if key is None:
            self.overlay_frame(frame_img, grid_width=grid_width, grid_height=grid_height, tile_size=tile_size)
            return

        if self._cache_one_frame:
            if self._one_frame_key == key and self._one_frame_img is not None:
                frame_img[:, :, :] = self._one_frame_img
                return
            drawn = np.zeros_like(frame_img)
            self.overlay_frame(drawn, grid_width=grid_width, grid_height=grid_height, tile_size=tile_size)
            self._one_frame_key = key
            self._one_frame_img = drawn
            frame_img[:, :, :] = drawn
            return

        cached = self._frame_cache.get(key)
        if cached is not None:
            frame_img[:, :, :] = cached
            return

        drawn = np.zeros_like(frame_img)
        self.overlay_frame(drawn, grid_width=grid_width, grid_height=grid_height, tile_size=tile_size)
        self._frame_cache[key] = drawn
        frame_img[:, :, :] = drawn

    # ------------------------------------------------------------------
    # overlay
    # ------------------------------------------------------------------
    @abstractmethod
    def overlay_tile(self, tile_img: np.ndarray, *, i: int, j: int, tile_size: int) -> None: ...
    
    def overlay_frame(
        self,
        frame_img: np.ndarray,
        *,
        grid_width: int,
        grid_height: int,
        tile_size: int,
    ) -> None:
        for j in range(grid_height):
            y0 = j * tile_size
            y1 = (j + 1) * tile_size
            for i in range(grid_width):
                x0 = i * tile_size
                x1 = (i + 1) * tile_size
                tile_view = frame_img[y0:y1, x0:x1, :]
                self.render_tile(tile_view, i=i, j=j, tile_size=tile_size)


class UnitRegisterInterface(ABC):
    @abstractmethod
    def register_engine(self, engine: "OverlayEngine") -> None: ...


class OverlayEngine:
    """Overlay render registry and execution engine (layer-ordered + enabled)."""

    def __init__(self) -> None:
        self._enabled_by_layer: dict[int, bool] = {}
        self._render_by_layer: dict[int, Any] = {}
        self._render_by_order: list[Any] = []
        self._profile_ns_by_layer: dict[int, int] = {}

    def register(self, layer: int, render: Any, *, enabled: bool = True) -> None:
        tile_fn = getattr(render, "overlay_tile", None)
        if not callable(tile_fn):
            raise TypeError("render must provide callable overlay_tile(tile_img, *, i, j, tile_size)")
        if layer in self._render_by_layer:
            raise ValueError(f"overlay layer already registered: {layer}")
        self._render_by_layer[layer] = render
        self._enabled_by_layer[layer] = enabled
        self._profile_ns_by_layer[layer] = 0
        self._rebuild_overlay_order()

    def set_enabled(self, layer: int, enabled: bool) -> None:
        if layer not in self._render_by_layer:
            raise KeyError(f"overlay layer not registered: {layer}")
        self._enabled_by_layer[layer] = enabled
        self._rebuild_overlay_order()

    def is_enabled(self, layer: int) -> bool:
        if layer not in self._render_by_layer:
            raise KeyError(f"overlay layer not registered: {layer}")
        return self._enabled_by_layer[layer]

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------
    def render_over_tile(self, *, grid_width: int, grid_height: int, tile_size: int) -> np.ndarray:
        frame_img = np.zeros((grid_height * tile_size, grid_width * tile_size, 3), dtype=np.uint8)
        for j in range(grid_height):
            y0 = j * tile_size
            y1 = (j + 1) * tile_size
            for i in range(grid_width):
                x0 = i * tile_size
                x1 = (i + 1) * tile_size
                tile_img = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                for layer, render in self._iter_enabled_renders():
                    t0 = time.perf_counter_ns()
                    render.render_tile(tile_img, i=i, j=j, tile_size=tile_size)
                    self._profile_ns_by_layer[layer] += time.perf_counter_ns() - t0
                frame_img[y0:y1, x0:x1, :] = tile_img
        return frame_img

    def render_over_frame(self, *, grid_width: int, grid_height: int, tile_size: int) -> np.ndarray:
        frame_img = np.zeros((grid_height * tile_size, grid_width * tile_size, 3), dtype=np.uint8)
        for layer, render in self._iter_enabled_renders():
            frame_fn = getattr(render, "render_frame", None)
            if callable(frame_fn):
                t0 = time.perf_counter_ns()
                frame_fn(frame_img, grid_width=grid_width, grid_height=grid_height, tile_size=tile_size)
                self._profile_ns_by_layer[layer] += time.perf_counter_ns() - t0
                continue
            t0 = time.perf_counter_ns()
            self._apply_tile_loop(render.render_tile, frame_img, grid_width=grid_width, grid_height=grid_height, tile_size=tile_size)
            self._profile_ns_by_layer[layer] += time.perf_counter_ns() - t0
        return frame_img

    # ------------------------------------------------------------------
    # overlay
    # ------------------------------------------------------------------
    def reset_profile(self) -> None:
        for layer in self._profile_ns_by_layer:
            self._profile_ns_by_layer[layer] = 0

    def get_profile_ms_by_layer(self) -> dict[int, float]:
        return {layer: ns / 1_000_000.0 for layer, ns in self._profile_ns_by_layer.items()}

    def get_profile_total_ms(self) -> float:
        return sum(self._profile_ns_by_layer.values()) / 1_000_000.0

    # ------------------------------------------------------------------
    # cache key
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_tile_loop(
        tile_fn: Callable[..., None],
        frame_img: np.ndarray,
        *,
        grid_width: int,
        grid_height: int,
        tile_size: int,
    ) -> None:
        for j in range(grid_height):
            y0 = j * tile_size
            y1 = (j + 1) * tile_size
            for i in range(grid_width):
                x0 = i * tile_size
                x1 = (i + 1) * tile_size
                tile_view = frame_img[y0:y1, x0:x1, :]
                tile_fn(tile_view, i=i, j=j)

    def _rebuild_overlay_order(self) -> None:
        self._render_by_order = [
            render
            for layer, render in sorted(self._render_by_layer.items(), key=lambda item: item[0])
            if self._enabled_by_layer.get(layer, True)
        ]

    def _iter_enabled_renders(self) -> list[tuple[int, Any]]:
        return [
            (layer, render)
            for layer, render in sorted(self._render_by_layer.items(), key=lambda item: item[0])
            if self._enabled_by_layer.get(layer, True)
        ]
