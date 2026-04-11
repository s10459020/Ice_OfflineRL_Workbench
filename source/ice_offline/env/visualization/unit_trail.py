from typing import Any

import numpy as np
from minigrid.utils.rendering import fill_coords, point_in_triangle, rotate_fn

from ice_offline.data.trail import Trail, TrailPoint
from ice_offline.dataset.state_collector import StateCollector
from ice_offline.dataset.state_loader import StateLoader
from .overlay_engine import OverlayEngine, RenderLayer
from .overlay_loader import UnitLoaderInterface
from .overlay_renderer import UnitRenderer
from .overlay_wrapper import UnitWrapperInterface


class TrailUnit(UnitWrapperInterface, UnitLoaderInterface, UnitRenderer):
    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def __init__(self, *, max_trails: int = 64, trail_mode: str = "rollout") -> None:
        UnitRenderer.__init__(self)
        if max_trails < 1:
            raise ValueError("max_trails must be >= 1")
        if trail_mode not in {"rollout", "clear"}:
            raise ValueError("trail_mode must be 'rollout' or 'clear'")

        self._trail = Trail(max_trails=max_trails)
        self._trail_mode = trail_mode

        self._trail_color = np.asarray((70, 160, 255), dtype=np.float32)
        self._arrow_masks_cache: dict[int, dict[int, np.ndarray]] = {}

    # ====================
    # Wrapper Hooks
    # ====================
    def on_wrapper(self, env: Any, wrapper: Any, engine: OverlayEngine):
        engine.register(int(RenderLayer.TRAIL), self)
        collector = StateCollector(env)
        wrapper.register("state", collector.get_last)
        return collector

    def on_reset(self, data: dict[str, Any]) -> None:
        state = data["state"]
        self._trail.reset()
        point = TrailPoint(pos=(int(state.agent_pos[0]), int(state.agent_pos[1])), direction=int(state.agent_dir))
        self._trail.push(point.pos, point.direction)

    def on_step(self, data: dict[str, Any]) -> None:
        state = data["state"]
        point = TrailPoint(pos=(int(state.agent_pos[0]), int(state.agent_pos[1])), direction=int(state.agent_dir))
        self._trail.push(point.pos, point.direction)

    # ====================
    # Loader Hooks
    # ====================
    def on_loader(self, engine: OverlayEngine, loader: Any) -> None:
        engine.register(int(RenderLayer.TRAIL), self)
        state_loader = StateLoader(loader.dataset_id)
        loader.register_list("state", lambda episode_index: state_loader.load_episode(episode_index))

    def on_load(self, datas: list[dict[str, Any]]) -> None:
        self._trail.reset()
        for data in datas:
            state = data["state"]
            point = TrailPoint(pos=(int(state.agent_pos[0]), int(state.agent_pos[1])), direction=int(state.agent_dir))
            self._trail.push(point.pos, point.direction)
        self._trail.set_view_end(0)

    def on_seek(self, data: dict[str, Any]) -> None:
        self._trail.set_view_end(data["step_index"])

    # ====================
    # Shared Hooks
    # ====================
    def on_render(self, _data: dict[str, Any]) -> None:
        if self._trail_mode == "clear":
            self._trail.reset()

    # ------------------------------------------------------------------
    # Cache Hooks
    # ------------------------------------------------------------------
    def condition_tile(self, *, i: int, j: int, tile_size: int) -> bool:
        return bool(self._trail.get_cell(i, j))

    def cache_tile_key(self, *, i: int, j: int, tile_size: int) -> int:
        entries = self._trail.get_cell(i, j)
        h = 0
        for level, trail_dir in entries:
            token = (((int(level) & 0x7) << 2) | (int(trail_dir) & 0x3)) + 1
            h = (h * 33) + token
        return h

    # ------------------------------------------------------------------
    # Render Hooks
    # ------------------------------------------------------------------
    def overlay_tile(self, tile_img: np.ndarray, *, i: int, j: int, tile_size: int) -> None:
        entries = self._trail.get_cell(i, j)
        overlay_rgb = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
        overlay_alpha = np.zeros((tile_size, tile_size), dtype=np.float32)
        arrow_masks = self._get_arrow_masks(tile_size)
        alpha_by_bucket = {1: 0.20, 2: 0.40, 3: 0.60, 4: 0.80}

        for bucket, trail_dir in entries:
            alpha = alpha_by_bucket[int(bucket)]
            mask = arrow_masks[trail_dir]
            overlay_rgb[mask] = self._trail_color
            overlay_alpha[mask] = alpha

        alpha_map = overlay_alpha[..., None]
        inv_alpha_map = 1.0 - alpha_map

        blended = inv_alpha_map * tile_img.astype(np.float32) + alpha_map * overlay_rgb
        tile_img[:, :, :] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    def _get_arrow_masks(self, tile_size: int) -> dict[int, np.ndarray]:
        arrow_masks = self._arrow_masks_cache.get(tile_size)
        if arrow_masks is not None:
            return arrow_masks

        arrow_masks = {}
        for d in range(4):
            ghost = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            tri_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * np.pi * d)
            fill_coords(ghost, tri_fn, (255, 255, 255))
            arrow_masks[d] = ghost[:, :, 0] > 0

        self._arrow_masks_cache[tile_size] = arrow_masks
        return arrow_masks
