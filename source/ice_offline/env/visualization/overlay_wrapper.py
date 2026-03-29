from types import MethodType
from typing import Any

import gymnasium as gym
from ice_offline.env.common.state_io_wrapper import ensure_state_io
import numpy as np
from .overlay_engine import (
    OverlayEngine,
)
from .overlay_engine import UnitRegisterInterface


class UnitWrapperInterface:
    """Unit base that can be attached to an OverlayWrapper."""

    def on_env(self, env: gym.Env) -> None:
        pass

    def on_reset(self, state: Any) -> None:
        pass

    def on_step(self, state: Any, action: Any, step_out: Any) -> None:
        pass

    def on_render(self, state: Any) -> None:
        pass


class OverlayWrapper(gym.Wrapper):
    """
    Overlay pipeline for MiniGrid tile rendering.

    Flow:
    1) Patch `grid.render` once per reset.
    2) Build each tile by ordered overlay callbacks.
    3) Apply overlays in sorted order (layer, id).
    """

    def __init__(self, env: gym.Env, units: list[Any] | None = None) -> None:
        units = list(units or [])
        for unit in units:
            if not isinstance(unit, UnitWrapperInterface) or not isinstance(unit, UnitRegisterInterface):
                raise TypeError("each unit must implement UnitWrapperInterface and UnitRegisterInterface")

        env = ensure_state_io(env)
        super().__init__(env)
        self._base_env = self.env.unwrapped
        self._get_state = self.env.env.get_state

        self.engine = OverlayEngine()
        self._units: list[Any] = units
        for unit in self._units:
            unit.on_env(self._base_env)
            unit.register_engine(self.engine)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._patch_get_frame()  # MiniGrid may recreate internals on reset.
        state = self._get_state()
        for unit in self._units:
            unit.on_reset(state)
        return out

    def step(self, action):
        out = self.env.step(action)
        state = self._get_state()
        for unit in self._units:
            unit.on_step(state, action, out)
        return out

    def render(self):
        state = self._get_state()
        for unit in self._units:
            unit.on_render(state)
        return self.env.render()

    # ------------------------------------------------------------------
    # Patched MiniGridEnv.get_frame callback
    # ------------------------------------------------------------------
    def _patch_get_frame(self) -> None:
        self._base_env.get_frame = MethodType(self._overlay_get_frame, self._base_env)

    def _overlay_get_frame(
        self,
        env_self: Any,
        highlight: bool = True,
        tile_size: int = 32,
        agent_pov: bool = False,
    ) -> np.ndarray:
        state = self._get_state()
        for unit in self._units:
            unit.on_render(state)

        return self.engine.render_over_tile(
            grid_width=int(env_self.width),
            grid_height=int(env_self.height),
            tile_size=tile_size,
        )

