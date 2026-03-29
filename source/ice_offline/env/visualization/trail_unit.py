from typing import Any

import gymnasium as gym
import numpy as np

from ..model.trail import Trail
from .overlay_engine import OverlayEngine, RenderLayer
from .overlay_engine import UnitRegisterInterface
from .overlay_engine import UnitRenderInterface
from .overlay_wrapper import UnitWrapperInterface
from .trail_render import TrailRenderer


class TrailUnit(UnitWrapperInterface, UnitRegisterInterface, UnitRenderInterface):
    def __init__(self, *, max_trails: int = 64, trail_mode: str = "rollout") -> None:
        super().__init__()
        if max_trails < 1:
            raise ValueError("max_trails must be >= 1")
        if trail_mode not in {"rollout", "clear"}:
            raise ValueError("trail_mode must be 'rollout' or 'clear'")
        
        self._trail = Trail(max_trails=max_trails)
        self._trail_mode = trail_mode
        self._engine: OverlayEngine | None = None
        self._renderer: TrailRenderer | None = None

    def register_engine(self, engine: OverlayEngine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Wrapper Hooks
    # ------------------------------------------------------------------
    def on_env(self, env: gym.Env) -> None:
        self._renderer = TrailRenderer(grid_width=env.unwrapped.width, grid_height=env.unwrapped.height, tile_size=env.unwrapped.tile_size)
        if self._engine is not None:
            self._engine.register(int(RenderLayer.TRAIL), self)

    def on_reset(self, state: Any) -> None:
        self._trail.reset()
        self._push_state(state)

    def on_step(self, state: Any, action: Any, step_out: Any) -> None:
        self._push_state(state)

    def on_render(self, state: Any) -> None:
        if self._trail_mode == "clear":
            self._trail.reset()

    # ------------------------------------------------------------------
    # Render Unit
    # ------------------------------------------------------------------
    def overlay_tile(self, tile_img: np.ndarray, *, i: int, j: int) -> None:
        if self._renderer is None:
            return
        self._renderer.overlay_tile(tile_img, trail=self._trail, i=i, j=j)

    def overlay_frame(
        self,
        frame_img: np.ndarray,
        *,
        grid_width: int,
        grid_height: int,
        tile_size: int,
    ) -> None:
        if self._renderer is None:
            return
        self._renderer.overlay_frame(frame_img, trail=self._trail)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _push_state(self, state: Any) -> None:
        pos = getattr(state, "agent_pos", None)
        direction = getattr(state, "agent_dir", None)
        if pos is None or direction is None:
            return
        self._trail.push((int(pos[0]), int(pos[1])), int(direction))
