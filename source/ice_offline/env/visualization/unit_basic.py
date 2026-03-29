import math

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
from ice_offline.env.model.state import State

from .overlay_engine import OverlayEngine, RenderLayer
from .overlay_engine import UnitRegisterInterface
from .overlay_engine import UnitRenderInterface
from .overlay_wrapper import UnitWrapperInterface


class _BackgroundRender(UnitRenderInterface):
    def __init__(self) -> None:
        self._cached_key: tuple[int, int, int] = (0, 0, 0)
        self._cached_frame: np.ndarray = np.zeros(0)

    def overlay_tile(self, tile_img: np.ndarray, *, i: int, j: int) -> None:
        fill_coords(tile_img, point_in_rect(0.0, 0.031, 0.0, 1.0), (100, 100, 100))
        fill_coords(tile_img, point_in_rect(0.0, 1.0, 0.0, 0.031), (100, 100, 100))

    def overlay_frame(self, frame_img: np.ndarray, *, w: int, h: int, tile_size: int) -> None:
        key = (w, h, tile_size)
        if self._cached_key == key:
            frame_img[:, :, :] = self._cached_frame
            return

        cached = np.zeros((h * tile_size, w * tile_size, 3), dtype=np.uint8)
        super().overlay_frame(cached, w, h, tile_size)
        self._cached_key = key
        self._cached_frame = cached
        frame_img[:, :, :] = cached


class _ObjectsRender(UnitRenderInterface):
    def __init__(self) -> None:
        self.grid: Grid | None = None

    def overlay_tile(self, tile_img: np.ndarray, *, i: int, j: int) -> None:
        if self.grid is None:
            return

        cell = self.grid.get(i, j)
        if cell is None:
            return
        cell.render(tile_img)


class _AgentRender(UnitRenderInterface):
    def __init__(self) -> None:
        self.agent_pos: tuple[int, int] = (0, 0)
        self.agent_dir: int = 0

    def overlay_tile(self, tile_img: np.ndarray, *, i: int, j: int) -> None:
        if self.agent_pos != (i, j):
            return
        tri_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.agent_dir)
        fill_coords(tile_img, tri_fn, (255, 0, 0))

    def overlay_frame(self, frame_img: np.ndarray, *, grid_width: int, grid_height: int, tile_size: int) -> None:
        i, j = self.agent_pos
        y0, y1 = j * tile_size, (j + 1) * tile_size
        x0, x1 = i * tile_size, (i + 1) * tile_size
        self.overlay_tile(frame_img[y0:y1, x0:x1, :], i=i, j=j)


class _HighlightRender(UnitRenderInterface):
    def __init__(self) -> None:
        self._env: gym.Env | None = None
        self.highlight_mask: np.ndarray | None = None

    def refresh_mask(self) -> None:
        env = self._env

        _, vis_mask = env.gen_obs_grid()
        f_vec = env.dir_vec
        r_vec = env.right_vec
        top_left = env.agent_pos + f_vec * (env.agent_view_size - 1) - r_vec * (env.agent_view_size // 2)

        mask = np.zeros((env.width, env.height), dtype=bool)
        for vis_j in range(env.agent_view_size):
            for vis_i in range(env.agent_view_size):
                if not vis_mask[vis_i, vis_j]:
                    continue
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                if 0 <= abs_i < env.width and 0 <= abs_j < env.height:
                    mask[abs_i, abs_j] = True
        self.highlight_mask = mask

    def overlay_tile(self, tile_img: np.ndarray, *, i: int, j: int) -> None:
        if self.highlight_mask is None:
            return
        if self.highlight_mask[i, j]:
            highlight_img(tile_img)


class BasicUnit(UnitWrapperInterface, UnitRegisterInterface):
    def __init__(self) -> None:
        self._agent = _AgentRender()
        self._objects = _ObjectsRender()
        self._highlight = _HighlightRender()
        self._background = _BackgroundRender()

    def register_engine(self, engine: OverlayEngine) -> None:
        engine.register(int(RenderLayer.AGENT), self._agent)
        engine.register(int(RenderLayer.OBJECTS), self._objects)
        engine.register(int(RenderLayer.HIGHLIGHT), self._highlight)
        engine.register(int(RenderLayer.BACKGROUND), self._background)

    def on_env(self, env: gym.Env) -> None:
        self._highlight._env = env

    def on_render(self, state: State) -> None:
        self._agent.agent_pos = state.agent_pos
        self._agent.agent_dir = state.agent_dir

        grid, _ = Grid.decode(np.asarray(state.grid, dtype=np.int16))
        self._objects.grid = grid
        
        self._highlight.refresh_mask()
