from enum import IntEnum
from types import MethodType
from typing import Any
import numpy as np
from ice_offline.tools.timing import Timer
from .overlay_renderer import OverlayRenderer


# ------------------------------------------------------------------
# Render Layers
# ------------------------------------------------------------------
class RenderLayer(IntEnum):
    BACKGROUND = 0
    DISTRIBUTION = 10
    OBJECTS = 20
    AGENT = 30
    HIGHLIGHT = 40


class OverlayEngine:
    """Overlay render registry and execution engine (layer-ordered + enabled)."""

    def __init__(self, *, base_env: Any, overlay_mode: str = "tile") -> None:
        self._enabled_by_layer: dict[int, bool] = {}
        self._render_by_layer: dict[int, Any] = {}
        self._render_by_order: list[Any] = []
        self._renderer = OverlayRenderer()
        self._patch_get_frame(base_env, overlay_mode=overlay_mode)

    def register(self, layer: int, render: Any, *, enabled: bool = True) -> None:
        tile_fn = getattr(render, "overlay_tile", None)
        if not callable(tile_fn):
            raise TypeError("render must provide callable overlay_tile(tile_img, *, i, j, tile_size)")
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

    def _rebuild_overlay_order(self) -> None:
        self._render_by_order = [
            render
            for layer, render in sorted(self._render_by_layer.items(), key=lambda item: item[0])
            if self._enabled_by_layer.get(layer, True)
        ]

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------
    def render_over_tile(self, *, grid_width: int, grid_height: int, tile_size: int) -> np.ndarray:
        renders_by_layer = self._iter_enabled_renders()
        renders = [render for _, render in renders_by_layer]
        frame_img, timing = self._renderer.render_over_tile(
            renders,
            grid_width=grid_width,
            grid_height=grid_height,
            tile_size=tile_size,
        )
        for layer in RenderLayer:
            Timer.set(f"overlay.layer.{layer.name.lower()}", 0.0)
        for (layer, _), elapsed in zip(renders_by_layer, timing):
            Timer.set(f"overlay.layer.{RenderLayer(layer).name.lower()}", elapsed / 1_000_000.0)
        return frame_img

    def render_over_frame(self, *, grid_width: int, grid_height: int, tile_size: int) -> np.ndarray:
        renders_by_layer = self._iter_enabled_renders()
        renders = [render for _, render in renders_by_layer]
        frame_img, timing = self._renderer.render_over_frame(
            renders,
            grid_width=grid_width,
            grid_height=grid_height,
            tile_size=tile_size,
        )
        for layer in RenderLayer:
            Timer.set(f"overlay.layer.{layer.name.lower()}", 0.0)
        for (layer, _), elapsed in zip(renders_by_layer, timing):
            Timer.set(f"overlay.layer.{RenderLayer(layer).name.lower()}", elapsed / 1_000_000.0)
        return frame_img

    def _iter_enabled_renders(self) -> list[tuple[int, Any]]:
        return [
            (layer, render)
            for layer, render in sorted(self._render_by_layer.items(), key=lambda item: item[0])
            if self._enabled_by_layer.get(layer, True)
        ]

    # ------------------------------------------------------------------
    # patch
    # ------------------------------------------------------------------
    def _overlay_get_frame_over_tile(
        self,
        env_self: Any,
        _highlight: bool = True,
        tile_size: int = 7,
        _agent_pov: bool = False,
    ) -> np.ndarray:
        return self.render_over_tile(
            grid_width=int(env_self.width),
            grid_height=int(env_self.height),
            tile_size=tile_size,
        )

    def _overlay_get_frame_over_frame(
        self,
        env_self: Any,
        _highlight: bool = True,
        tile_size: int = 7,
        _agent_pov: bool = False,
    ) -> np.ndarray:
        return self.render_over_frame(
            grid_width=int(env_self.width),
            grid_height=int(env_self.height),
            tile_size=tile_size,
        )

    def _patch_get_frame(self, base_env: Any, *, overlay_mode: str = "tile") -> None:
        if overlay_mode == "tile":
            base_env.get_frame = MethodType(self._overlay_get_frame_over_tile, base_env)
            return
        if overlay_mode == "frame":
            base_env.get_frame = MethodType(self._overlay_get_frame_over_frame, base_env)
            return
        raise ValueError("overlay_mode must be 'tile' or 'frame'")
