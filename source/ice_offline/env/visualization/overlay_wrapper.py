from typing import Any

import gymnasium as gym
from ice_offline.env.common.state_io_wrapper import ensure_state_io
from ice_offline.env.model import State
from .overlay_engine import (
    OverlayEngine,
)
from .overlay_engine import UnitRegisterInterface


class UnitWrapperInterface:
    """Unit base that can be attached to an OverlayWrapper."""

    def on_env(self, env: gym.Env) -> None:
        pass

    def on_reset(self, state: State) -> None:
        pass

    def on_step(self, state: State, action: Any, step_out: Any) -> None:
        pass

    def on_render(self, state: State) -> None:
        pass


class OverlayWrapper(gym.Wrapper):
    """
    Overlay pipeline for MiniGrid tile rendering.

    Flow:
    1) Patch `grid.render` once per reset.
    2) Build each tile by ordered overlay callbacks.
    3) Apply overlays in sorted order (layer, id).
    """

    def __init__(self, env: gym.Env, units: list[Any] | None = None, *, overlay_mode: str = "tile") -> None:
        units = list(units or [])
        for unit in units:
            if not isinstance(unit, UnitWrapperInterface) or not isinstance(unit, UnitRegisterInterface):
                raise TypeError("each unit must implement UnitWrapperInterface and UnitRegisterInterface")

        env = ensure_state_io(env)
        super().__init__(env)
        self._base_env = self.env.unwrapped
        self._get_state = self.env.env.get_state

        self.engine = OverlayEngine(base_env=self._base_env, overlay_mode=overlay_mode)
        self._units: list[Any] = units
        for unit in self._units:
            unit.on_env(self._base_env)
            unit.register_engine(self.engine)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
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
