from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym

from .state_writer import StateDatasetWriter


class StateCollector:
    """Collect state trajectories into state_dataset_v1."""

    def __init__(self, output_path: str | Path, write_interval: int = 0) -> None:
        self.output_path = Path(output_path)
        self._writer = StateDatasetWriter(output_path=self.output_path, write_interval=write_interval)

    def prepare_env(self, env: gym.Env) -> gym.Env:
        return self._writer.wrap_env(env)

    def on_reset(self, info: dict[str, Any]) -> None:
        self._writer.on_reset(info)

    def on_step(
        self,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self._writer.on_step(action, reward, terminated, truncated, info)

    def on_episode_end(self, *, forced_cutoff: bool) -> None:
        if forced_cutoff:
            self._writer.end_episode()

    def close(self) -> None:
        self._writer.close()

    def result(self) -> dict[str, Any]:
        return {"path": str(self.output_path)}
