import math

import numpy as np
from minigrid.utils.rendering import fill_coords, point_in_triangle, rotate_fn

from ..model.trail import Trail


class TrailRenderer:
    """Trail drawing component."""

    def __init__(
        self,
        *,
        grid_width: int,
        grid_height: int,
        tile_size: int,
    ) -> None:
        self._trail_color = np.asarray((70, 160, 255), dtype=np.float32)
        
        # Max entries = 1 in this renderer instance because tile_size is fixed after init.
        self._arrow_masks_cache: dict[int, dict[int, np.ndarray]] = {}
        # Max entries = 7,888: keep at most 4 directions, each has 4 levels, and order matters.
        self.tile_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._grid_width = int(grid_width)
        self._grid_height = int(grid_height)
        self._tile_size = int(tile_size)

    def overlay_tile(self, tile_img: np.ndarray, *, trail: Trail, i: int, j: int) -> None:
        entries = trail.get_cell(i, j)
        if not entries:
            return

        key = trail.get_cell_hash(i, j)
        cached = self.tile_cache.get(key)
        if cached is not None:
            overlay_rgb, alpha_map, inv_alpha_map = cached
        else:
            tile_size = self._tile_size
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
            self.tile_cache[key] = (overlay_rgb, alpha_map, inv_alpha_map)

        blended = inv_alpha_map * tile_img.astype(np.float32) + alpha_map * overlay_rgb
        tile_img[:, :, :] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    def overlay_frame(self, frame_img: np.ndarray, *, trail: Trail) -> None:
        tile_size = self._tile_size
        for j in range(self._grid_height):
            y0 = j * tile_size
            y1 = (j + 1) * tile_size
            for i in range(self._grid_width):
                x0 = i * tile_size
                x1 = (i + 1) * tile_size
                tile_view = frame_img[y0:y1, x0:x1, :]
                self.overlay_tile(tile_view, trail=trail, i=i, j=j)

    def _get_arrow_masks(self, tile_size: int) -> dict[int, np.ndarray]:
        arrow_masks = self._arrow_masks_cache.get(tile_size)
        if arrow_masks is not None:
            return arrow_masks

        arrow_masks = {}
        for d in range(4):
            ghost = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            tri_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * d)
            fill_coords(ghost, tri_fn, (255, 255, 255))
            arrow_masks[d] = ghost[:, :, 0] > 0

        self._arrow_masks_cache[tile_size] = arrow_masks
        return arrow_masks
