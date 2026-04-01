from typing import Any

import gymnasium as gym
from ice_offline.env.common.state_io_wrapper import StateIOWrapper
from ice_offline.env.model import State
from .overlay_engine import (
    OverlayEngine,
)
from .overlay_engine import UnitRegisterInterface


# ------------------------------------------------------------------
# Unit Interface
# ------------------------------------------------------------------
class UnitWrapperInterface:
    def on_wrapper(self, env: gym.Env) -> gym.Env:
        return env

    def on_env(self, base_env: gym.Env) -> None:
        pass

    def on_reset(self, state: State, info: dict[str, Any]) -> None:
        pass

    def on_step(self, state: State, action: Any, reward: float, done: bool, info: dict[str, Any]) -> None:
        pass

    def on_render(self, state: State, info: dict[str, Any]) -> None:
        pass


class OverlayWrapper(gym.Wrapper):
    """
    Overlay pipeline for MiniGrid tile rendering.

    Flow:
    1) Patch `grid.render` once per reset.
    2) Build each tile by ordered overlay callbacks.
    3) Apply overlays in sorted order (layer, id).
    """

    def __init__(self, env: gym.Env, units: list[Any]) -> None:
        for unit in units:
            if not isinstance(unit, UnitWrapperInterface) or not isinstance(unit, UnitRegisterInterface):
                raise TypeError("each unit must implement UnitWrapperInterface and UnitRegisterInterface")

        for unit in units:
            env = unit.on_wrapper(env)

        self._state_io = StateIOWrapper(env)
        super().__init__(self._state_io)

        self.engine = OverlayEngine(base_env=self.env.unwrapped, overlay_mode="tile")
        self._units: list[Any] = units
        self._last_state: State | None = None
        self._last_info: dict[str, Any] = {}
        
        for unit in self._units:
            unit.on_env(self.env.unwrapped)
            unit.register_engine(self.engine)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = self._state_io.get_state()

        self._last_state = state
        self._last_info = dict(info)

        for unit in self._units:
            unit.on_reset(state, dict(info))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        state = self._state_io.get_state()

        self._last_state = state
        self._last_info = dict(info)

        for unit in self._units:
            unit.on_step(state, action, float(reward), done, dict(info))
        return obs, reward, terminated, truncated, info

    def render(self):
        for unit in self._units:
            unit.on_render(self._last_state, self._last_info)
        return self.env.render()
