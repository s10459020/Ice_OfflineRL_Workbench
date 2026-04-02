from typing import Any

import gymnasium as gym
from ice_offline.env.common import insert_render_quiet_innermost
from .overlay_engine import OverlayEngine


# ------------------------------------------------------------------
# Unit Interface
# ------------------------------------------------------------------
class UnitWrapperInterface:
    # ====================
    # Wrapper Lifecycle
    # Order: on_wrapper -> on_env -> on_reset -> on_step -> on_render
    # ====================
    def on_wrapper(self, env: gym.Env, wrapper: "OverlayWrapper", engine: OverlayEngine) -> gym.Env:
        return env

    def on_env(self, base_env: gym.Env) -> None:
        pass

    def on_reset(self, data: dict[str, Any]) -> None:
        pass

    def on_step(self, data: dict[str, Any]) -> None:
        pass

    def on_render(self, data: dict[str, Any]) -> None:
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
            if not isinstance(unit, UnitWrapperInterface):
                raise TypeError("each unit must implement UnitWrapperInterface")

        env = insert_render_quiet_innermost(env)
        self.engine = OverlayEngine(base_env=env.unwrapped, overlay_mode="tile")
        self.data: dict[str, Any] = {}
        self._snapshots: dict[str, Any] = {}
        self._units: list[Any] = units

        for unit in units:
            env = unit.on_wrapper(env, self, self.engine)

        super().__init__(env)

        for unit in units:
            unit.on_env(self.env.unwrapped)

    def register(self, key: str, callback: Any) -> None:
        if key in self._snapshots:
            return
        self._snapshots[key] = callback

    def _refresh_snapshots(self) -> None:
        for key, callback in self._snapshots.items():
            self.data[key] = callback()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.data["observation"] = obs
        self.data["info"] = dict(info)
        self._refresh_snapshots()

        for unit in self._units:
            unit.on_reset(self.data)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        self.data["action"] = action
        self.data["reward"] = float(reward)
        self.data["terminated"] = bool(terminated)
        self.data["truncated"] = bool(truncated)
        self.data["done"] = done
        self.data["observation"] = obs
        self.data["info"] = dict(info)
        self._refresh_snapshots()

        for unit in self._units:
            unit.on_step(self.data)
        return obs, reward, terminated, truncated, info

    def render(self):
        for unit in self._units:
            unit.on_render(self.data)
        return self.env.render()
