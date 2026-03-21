import math
from collections.abc import Callable
from enum import IntEnum
from types import MethodType
from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.utils.rendering import (
    fill_coords,
    highlight_img,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)


class RenderLayer(IntEnum):
    BACKGROUND = 0
    OBJECTS = 10
    AGENT = 40
    HIGHLIGHT = 50


TileOverlayFn = Callable[[np.ndarray, dict[str, Any]], None]


class RenderOverlayWrapper(gym.Wrapper):
    """
    Overlay pipeline for MiniGrid tile rendering.

    Flow:
    1) Patch `grid.render` once per reset.
    2) Build each tile by ordered overlay callbacks.
    3) Apply overlays in sorted order (layer, id).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._base_env = self.env.unwrapped
        self._overlay_by_layer: dict[int, TileOverlayFn] = {}
        self._ordered_overlay_fns: list[TileOverlayFn] = []

        # Built-in overlays.
        self.register_overlay(RenderLayer.BACKGROUND, self._overlay_background)
        self.register_overlay(RenderLayer.OBJECTS, self._overlay_objects)
        self.register_overlay(RenderLayer.AGENT, self._overlay_agent)
        self.register_overlay(RenderLayer.HIGHLIGHT, self._overlay_highlight)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_overlay(
        self,
        layer: int,
        fn: TileOverlayFn,
    ) -> None:
        if layer in self._overlay_by_layer:
            raise ValueError(f"overlay layer already registered: {layer}")
        
        self._overlay_by_layer[layer] = fn
        # Overlay order only changes at register-time.
        self._ordered_overlay_fns = [
            overlay_fn
            for _, overlay_fn in sorted(self._overlay_by_layer.items(), key=lambda item: item[0])
        ]

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._patch_grid_render() # MiniGrid may recreate `grid` on reset.
        return out

    # ------------------------------------------------------------------
    # Render flow internals
    # ------------------------------------------------------------------
    def _patch_grid_render(self) -> None:
        grid = self._base_env.grid
        grid.render = MethodType(self._overlay_render, grid)

    def _build_tile_context(
        self,
        i: int,
        j: int,
        cell: Any,
        tile_size: int,
        agent_pos: tuple[int, int],
        agent_dir: int | None,
        highlight_mask: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "i": i,
            "j": j,
            "cell": cell,
            "tile_size": tile_size,
            "agent_pos": agent_pos,
            "agent_dir": agent_dir,
            "agent_here": np.array_equal(agent_pos, (i, j)),
            "highlight": highlight_mask[i, j],
        }

    # ------------------------------------------------------------------
    # Built-in overlays
    # ------------------------------------------------------------------
    def _overlay_background(self, tile_img: np.ndarray, _ctx: dict[str, Any]) -> None:
        # Match MiniGrid native tile base: black background + top/left borders.
        fill_coords(tile_img, point_in_rect(0.0, 0.031, 0.0, 1.0), (100, 100, 100))
        fill_coords(tile_img, point_in_rect(0.0, 1.0, 0.0, 0.031), (100, 100, 100))

    def _overlay_objects(self, tile_img: np.ndarray, ctx: dict[str, Any]) -> None:
        cell = ctx["cell"]
        if cell is None:
            return
        cell.render(tile_img)

    def _overlay_agent(self, tile_img: np.ndarray, ctx: dict[str, Any]) -> None:
        if not ctx["agent_here"] or ctx["agent_dir"] is None:
            return
        tri_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
        tri_fn = rotate_fn(
            tri_fn,
            cx=0.5,
            cy=0.5,
            theta=0.5 * math.pi * ctx["agent_dir"],
        )
        fill_coords(tile_img, tri_fn, (255, 0, 0))

    def _overlay_highlight(self, tile_img: np.ndarray, ctx: dict[str, Any]) -> None:
        if ctx["highlight"]:
            highlight_img(tile_img)

    # ------------------------------------------------------------------
    # Patched Grid.render callback
    # ------------------------------------------------------------------
    def _overlay_render(
        self,
        grid_self: Grid,
        tile_size: int,
        agent_pos: tuple[int, int],
        agent_dir: int | None = None,
        highlight_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if highlight_mask is None:
            highlight_mask = np.zeros((grid_self.width, grid_self.height), dtype=bool)

        overlays = self._ordered_overlay_fns
        img = np.zeros((grid_self.height * tile_size, grid_self.width * tile_size, 3), dtype=np.uint8)

        for j in range(grid_self.height):
            for i in range(grid_self.width):
                cell = grid_self.get(i, j)
                ctx = self._build_tile_context(
                    i=i,
                    j=j,
                    cell=cell,
                    tile_size=tile_size,
                    agent_pos=agent_pos,
                    agent_dir=agent_dir,
                    highlight_mask=highlight_mask,
                )
                tile_img = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

                for overlay_fn in overlays:
                    overlay_fn(tile_img, ctx)

                y0 = j * tile_size
                y1 = (j + 1) * tile_size
                x0 = i * tile_size
                x1 = (i + 1) * tile_size
                img[y0:y1, x0:x1, :] = tile_img

        return img


class OverlayDependentWrapper(gym.Wrapper):
    """
    Base wrapper for modules that depend on RenderOverlayWrapper.

    Common behavior:
    1) Reuse existing RenderOverlayWrapper if present.
    2) Auto-insert RenderOverlayWrapper when missing.
    3) Register callback and keep overlay id.
    """

    def __init__(
        self,
        env: gym.Env,
        overlay_fn: TileOverlayFn,
        overlay_layer: int,
    ) -> None:
        overlay_wrapper = self._find_overlay_wrapper(env)
        if overlay_wrapper is None:
            env = RenderOverlayWrapper(env)
            overlay_wrapper = env
        
        super().__init__(env)
        self._overlay_wrapper: RenderOverlayWrapper = overlay_wrapper
        self._overlay_wrapper.register_overlay(
            overlay_layer,
            overlay_fn,
        )

    @staticmethod
    def _find_overlay_wrapper(env: gym.Env) -> RenderOverlayWrapper | None:
        current: gym.Env = env
        while isinstance(current, gym.Wrapper):
            if isinstance(current, RenderOverlayWrapper):
                return current
            current = current.env
        return None
