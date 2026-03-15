from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from types import MethodType
from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.utils.rendering import fill_coords, highlight_img, point_in_triangle, rotate_fn


class RenderLayer(IntEnum):
    BACKGROUND = 0
    OBJECTS = 10
    DISTRIBUTION = 20
    TRAIL = 30
    AGENT = 40
    HIGHLIGHT = 50


TileOverlayFn = Callable[[np.ndarray, dict[str, Any]], None]


@dataclass
class _OverlaySpec:
    name: str
    fn: TileOverlayFn
    layer: int
    z: int
    enabled: bool
    order: int


class RenderOverlayWrapper(gym.Wrapper):
    """Patch Grid.render once and apply ordered tile overlays."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._base_env = self.env.unwrapped
        self._overlay_specs: dict[str, _OverlaySpec] = {}
        self._register_counter = 0
        self.register_overlay(
            name="agent",
            fn=self._overlay_agent,
            layer=int(RenderLayer.AGENT),
            z=0,
            enabled=True,
        )
        self.register_overlay(
            name="highlight",
            fn=self._overlay_highlight,
            layer=int(RenderLayer.HIGHLIGHT),
            z=0,
            enabled=True,
        )
        self._patch_grid_render()

    def register_overlay(
        self,
        name: str,
        fn: TileOverlayFn,
        *,
        layer: int,
        z: int = 0,
        enabled: bool = True,
    ) -> None:
        self._register_counter += 1
        self._overlay_specs[name] = _OverlaySpec(
            name=name,
            fn=fn,
            layer=int(layer),
            z=int(z),
            enabled=bool(enabled),
            order=self._register_counter,
        )

    def unregister_overlay(self, name: str) -> None:
        self._overlay_specs.pop(name, None)

    def set_overlay_enabled(self, name: str, enabled: bool) -> None:
        spec = self._overlay_specs.get(name)
        if spec is None:
            raise KeyError(f"overlay not found: {name}")
        spec.enabled = bool(enabled)

    def _sorted_overlays(self) -> list[_OverlaySpec]:
        overlays = [spec for spec in self._overlay_specs.values() if spec.enabled]
        overlays.sort(key=lambda s: (s.layer, s.z, s.order))
        return overlays

    def _patch_grid_render(self) -> None:
        grid = self._base_env.grid
        grid.render = MethodType(self._overlay_render, grid)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        # MiniGrid may recreate grid on reset.
        self._patch_grid_render()
        return out

    def _overlay_agent(self, tile_img: np.ndarray, ctx: dict[str, Any]) -> None:
        if not ctx["agent_here"] or ctx["agent_dir"] is None:
            return
        tri_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
        tri_fn = rotate_fn(
            tri_fn,
            cx=0.5,
            cy=0.5,
            theta=0.5 * math.pi * int(ctx["agent_dir"]),
        )
        fill_coords(tile_img, tri_fn, (255, 0, 0))

    def _overlay_highlight(self, tile_img: np.ndarray, ctx: dict[str, Any]) -> None:
        if bool(ctx["highlight"]):
            highlight_img(tile_img)

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

        overlays = self._sorted_overlays()
        width_px = grid_self.width * tile_size
        height_px = grid_self.height * tile_size
        img = np.zeros((height_px, width_px, 3), dtype=np.uint8)

        for j in range(grid_self.height):
            for i in range(grid_self.width):
                cell = grid_self.get(i, j)
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=None,
                    highlight=False,
                    tile_size=tile_size,
                ).astype(np.uint8)
                ctx = {
                    "i": i,
                    "j": j,
                    "cell": cell,
                    "tile_size": tile_size,
                    "agent_pos": agent_pos,
                    "agent_dir": agent_dir,
                    "agent_here": np.array_equal(agent_pos, (i, j)),
                    "highlight": bool(highlight_mask[i, j]),
                }

                for spec in overlays:
                    spec.fn(tile_img, ctx)

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img
