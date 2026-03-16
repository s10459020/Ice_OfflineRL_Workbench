from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class AgentState:
    mission: str
    agent_pos: tuple[int, int]
    agent_dir: int
    grid: np.ndarray
    carrying: dict[str, Any] | None


class StateCaptureWrapper(gym.Wrapper):
    """Attach render-relevant MiniGrid state to info on reset/step."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        state = self._capture_state()
        info = dict(info)
        info["state"] = state
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = self._capture_state()
        info = dict(info)
        info["state"] = state
        return obs, reward, terminated, truncated, info

    def _capture_state(self) -> AgentState:
        base = self.env.unwrapped
        x, y = base.agent_pos
        carrying = self._serialize_carrying(base.carrying)

        # Copy grid to snapshot the current world state at this exact step.
        grid = np.asarray(base.grid.encode(), dtype=np.uint8).copy()

        return AgentState(
            mission=str(base.mission),
            agent_pos=(int(x), int(y)),
            agent_dir=int(base.agent_dir),
            grid=grid,
            carrying=carrying,
        )

    @staticmethod
    def _serialize_carrying(carrying_obj: Any) -> dict[str, Any] | None:
        if carrying_obj is None:
            return None

        encoded = carrying_obj.encode()
        return {
            "type": str(carrying_obj.type),
            "color": str(carrying_obj.color),
            "state": int(encoded[2]),
            "encoded": tuple(int(v) for v in encoded),
        }
