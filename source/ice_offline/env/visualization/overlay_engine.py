from enum import IntEnum
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
import numpy as np


class RenderLayer(IntEnum):
    BACKGROUND = 0
    OBJECTS = 10
    TRAIL = 30
    AGENT = 40
    HIGHLIGHT = 50


class UnitRenderInterface(ABC):
    @abstractmethod
    def overlay_tile(self, tile_img: np.ndarray, *, i: int, j: int) -> None: ...

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
                self.overlay_tile(tile_view, i=i, j=j)


class UnitRegisterInterface(ABC):
    @abstractmethod
    def register_engine(self, engine: "OverlayEngine") -> None: ...


class OverlayEngine:
    """Overlay render registry and execution engine (layer-ordered + enabled)."""

    def __init__(self) -> None:
        self._enabled_by_layer: dict[int, bool] = {}
        self._render_by_layer: dict[int, Any] = {}
        self._render_by_order: list[Any] = []

    def register(self, layer: int, render: Any, *, enabled: bool = True) -> None:
        tile_fn = getattr(render, "overlay_tile", None)
        if not callable(tile_fn):
            raise TypeError("render must provide callable overlay_tile(tile_img, *, i, j)")
        if layer in self._render_by_layer:
            raise ValueError(f"overlay layer already registered: {layer}")
        self._render_by_layer[layer] = render
        self._enabled_by_layer[layer] = enabled
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

    def render_over_tile(self, *, grid_width: int, grid_height: int, tile_size: int) -> np.ndarray:
        frame_img = np.zeros((grid_height * tile_size, grid_width * tile_size, 3), dtype=np.uint8)
        for j in range(grid_height):
            y0 = j * tile_size
            y1 = (j + 1) * tile_size
            for i in range(grid_width):
                x0 = i * tile_size
                x1 = (i + 1) * tile_size
                tile_img = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                for render in self._render_by_order:
                    render.overlay_tile(tile_img, i=i, j=j)
                frame_img[y0:y1, x0:x1, :] = tile_img
        return frame_img

    def render_over_frame(self, *, grid_width: int, grid_height: int, tile_size: int) -> np.ndarray:
        frame_img = np.zeros((grid_height * tile_size, grid_width * tile_size, 3), dtype=np.uint8)
        for render in self._render_by_order:
            frame_fn = getattr(render, "overlay_frame", None)
            if callable(frame_fn):
                frame_fn(frame_img, grid_width=grid_width, grid_height=grid_height, tile_size=tile_size)
                continue
            self._apply_tile_loop(render.overlay_tile, frame_img, grid_width=grid_width, grid_height=grid_height, tile_size=tile_size)
        return frame_img

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

