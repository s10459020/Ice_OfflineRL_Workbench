from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym

from .state_capture_wrapper import StateCaptureWrapper
from .state_types import AgentState
from .write_state_dataset import StateDatasetWriter


class StateCollector:
    """Collect state trajectories into state_dataset_v1."""

    def __init__(
        self,
        output_path: str | Path,
        flush_interval: int = 0,
    ) -> None:
        self.output_path = Path(output_path)
        self._writer = StateDatasetWriter(output_path=self.output_path, flush_interval=flush_interval)
        self._episode_open = False

    def prepare_env(self, env: gym.Env) -> gym.Env:
        return ensure_state_capture(env)

    def on_reset(self, info: dict[str, Any]) -> None:
        self._writer.push_state(self._extract_state(info))
        self._episode_open = True

    def on_step(
        self,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        _ = (action, reward)
        self._writer.push_state(self._extract_state(info))
        if terminated or truncated:
            self._writer.end_episode()
            self._episode_open = False

    def on_episode_end(self, *, forced_cutoff: bool) -> None:
        if forced_cutoff and self._episode_open:
            self._writer.end_episode()
            self._episode_open = False

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()

    def result(self) -> dict[str, Any]:
        return {"path": str(self.output_path)}

    @staticmethod
    def _extract_state(info: dict[str, Any]) -> AgentState:
        state = info.get("state")
        if not isinstance(state, AgentState):
            raise KeyError("info['state'] missing. Wrap env with StateCaptureWrapper first.")
        return state


def ensure_state_capture(env: gym.Env) -> gym.Env:
    current = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, StateCaptureWrapper):
            return env
        current = current.env
    return StateCaptureWrapper(env)
