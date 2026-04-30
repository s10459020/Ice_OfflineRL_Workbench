from typing import Any

import numpy as np


class DistributionOverlayInterface:
    """Distribution overlay unit with explicit tile/frame interfaces."""

    def __init__(self, renderer: Any) -> None:
        self._renderer = renderer
        self._values: np.ndarray | None = None
        self._quantile_edges = np.zeros(4, dtype=np.float32)
        self._grid_width: int | None = None
        self._grid_height: int | None = None
        self._tile_size: int | None = None

    def set_grid_spec(self, grid_width: int, grid_height: int, tile_size: int) -> None:
        self._grid_width = int(grid_width)
        self._grid_height = int(grid_height)
        self._tile_size = int(tile_size)

    def set_frame_values(self, values: np.ndarray | None, quantile_edges: np.ndarray | None = None) -> None:
        self._values = values
        if quantile_edges is None:
            self._quantile_edges = np.zeros(4, dtype=np.float32)
        else:
            self._quantile_edges = np.asarray(quantile_edges, dtype=np.float32).reshape(4)

    def overlay_tile(self, tile_img: np.ndarray, ctx: dict[str, Any]) -> None:
        if self._values is None:
            return
        i = int(ctx["i"])
        j = int(ctx["j"])
        if i == 0 or j == 0:
            return
        ix = i - 1
        iy = j - 1
        if ix < 0 or iy < 0 or ix >= self._values.shape[0] or iy >= self._values.shape[1]:
            return
        cell_values = self._values[ix, iy]
        self._renderer.render(tile_img, cell_values, self._quantile_edges)

    def overlay_frame(self, frame_img: np.ndarray, ctx: dict[str, Any] | None = None) -> None:
        base_ctx = {} if ctx is None else dict(ctx)
        grid_width = self._grid_width
        grid_height = self._grid_height
        tile_size = self._tile_size

        if grid_width is None or grid_height is None or tile_size is None:
            raise ValueError("grid spec is not set. call set_grid_spec(...).")

        for j in range(grid_height):
            y0 = j * tile_size
            y1 = (j + 1) * tile_size
            for i in range(grid_width):
                x0 = i * tile_size
                x1 = (i + 1) * tile_size
                tile_view = frame_img[y0:y1, x0:x1, :]
                tile_ctx = dict(base_ctx)
                tile_ctx["i"] = i
                tile_ctx["j"] = j
                tile_ctx["tile_size"] = tile_size
                self.overlay_tile(tile_view, tile_ctx)
