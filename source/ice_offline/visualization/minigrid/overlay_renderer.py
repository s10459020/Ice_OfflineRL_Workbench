from abc import ABC, abstractmethod
import time
from typing import Any

import numpy as np


# ------------------------------------------------------------------
# Unit Render Interface + Cache Layer
# ------------------------------------------------------------------
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
    # Cache Hooks
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
    # Cached Render Entry Points
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
    # Overlay Draw Hooks
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


# ------------------------------------------------------------------
# Overlay Render Executor
# ------------------------------------------------------------------
class OverlayRenderer:
    """Low-level render executor for tile/frame overlay composition."""

    # ------------------------------------------------------------------
    # Tile Pass
    # ------------------------------------------------------------------
    def render_over_tile(
        self,
        renders: list[Any],
        *,
        grid_width: int,
        grid_height: int,
        tile_size: int,
    ) -> tuple[np.ndarray, list[int]]:
        timing_ns_list = [0 for _ in renders]
        frame_img = np.zeros((grid_height * tile_size, grid_width * tile_size, 3), dtype=np.uint8)
        for j in range(grid_height):
            y0 = j * tile_size
            y1 = (j + 1) * tile_size
            for i in range(grid_width):
                x0 = i * tile_size
                x1 = (i + 1) * tile_size
                tile_img = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                for idx, render in enumerate(renders):
                    t0 = time.perf_counter_ns()
                    render.render_tile(tile_img, i=i, j=j, tile_size=tile_size)
                    timing_ns_list[idx] += time.perf_counter_ns() - t0
                frame_img[y0:y1, x0:x1, :] = tile_img
        return frame_img, timing_ns_list

    # ------------------------------------------------------------------
    # Frame Pass
    # ------------------------------------------------------------------
    def render_over_frame(
        self,
        renders: list[Any],
        *,
        grid_width: int,
        grid_height: int,
        tile_size: int,
    ) -> tuple[np.ndarray, list[int]]:
        timing_ns_list = [0 for _ in renders]
        frame_img = np.zeros((grid_height * tile_size, grid_width * tile_size, 3), dtype=np.uint8)
        for idx, render in enumerate(renders):
            frame_fn = getattr(render, "render_frame", None)
            if callable(frame_fn):
                t0 = time.perf_counter_ns()
                frame_fn(frame_img, grid_width=grid_width, grid_height=grid_height, tile_size=tile_size)
                timing_ns_list[idx] += time.perf_counter_ns() - t0
                continue
            t0 = time.perf_counter_ns()
            for j in range(grid_height):
                y0 = j * tile_size
                y1 = (j + 1) * tile_size
                for i in range(grid_width):
                    x0 = i * tile_size
                    x1 = (i + 1) * tile_size
                    tile_view = frame_img[y0:y1, x0:x1, :]
                    render.render_tile(tile_view, i=i, j=j, tile_size=tile_size)
            timing_ns_list[idx] += time.perf_counter_ns() - t0
        return frame_img, timing_ns_list
