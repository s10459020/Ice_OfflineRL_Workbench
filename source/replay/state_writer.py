from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from .state_capture_wrapper import StateCaptureWrapper
from .state_types import AgentState

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    h5py = None
    _H5PY_IMPORT_ERROR = exc
else:
    _H5PY_IMPORT_ERROR = None


@dataclass
class _EpisodeBuffer:
    missions: list[str] = field(default_factory=list)
    agent_pos: list[tuple[int, int]] = field(default_factory=list)
    agent_dir: list[int] = field(default_factory=list)
    grid: list[np.ndarray] = field(default_factory=list)
    carrying: list[str] = field(default_factory=list)

    def append_state(self, state: AgentState) -> None:
        self.missions.append(str(state.mission))
        self.agent_pos.append((int(state.agent_pos[0]), int(state.agent_pos[1])))
        self.agent_dir.append(int(state.agent_dir))
        self.grid.append(np.asarray(state.grid, dtype=np.uint8).copy())
        self.carrying.append(json.dumps(state.carrying, ensure_ascii=True))

    @property
    def num_states(self) -> int:
        return len(self.missions)


def ensure_state_capture(env: gym.Env) -> gym.Env:
    current = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, StateCaptureWrapper):
            return env
        current = current.env
    return StateCaptureWrapper(env)


class StateDatasetWriter:
    """
    Write per-episode MiniGrid state trajectories to HDF5.

    write_interval:
      - 0: keep all completed episodes in memory and write once on close.
      - N (>0): flush completed episodes every N episodes.
    """

    def __init__(
        self,
        output_path: str | Path,
        write_interval: int = 0,
        compression: str | None = "gzip",
    ) -> None:
        if write_interval < 0:
            raise ValueError("write_interval must be >= 0.")
        if h5py is None:  # pragma: no cover
            raise ImportError("h5py is required for StateDatasetWriter.") from _H5PY_IMPORT_ERROR

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_interval = int(write_interval)
        self.compression = compression

        self._file = h5py.File(self.output_path, "w")
        self._file.attrs["format"] = "state_dataset_v1"
        self._file.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        self._file.attrs["write_interval"] = self.write_interval

        self._pending_episodes: list[_EpisodeBuffer] = []
        self._episode_index = 0
        self._current: _EpisodeBuffer | None = None
        self._closed = False

    def wrap_env(self, env: gym.Env) -> gym.Env:
        return ensure_state_capture(env)

    def on_reset(self, info: dict[str, Any]) -> None:
        self._ensure_open()
        if self._current is not None:
            raise RuntimeError("Previous episode not ended. Call end_episode() first.")
        state = self._extract_state(info)
        self._current = _EpisodeBuffer()
        self._current.append_state(state)

    def on_step(
        self,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self._ensure_open()
        if self._current is None:
            raise RuntimeError("Call on_reset() before on_step().")
        next_state = self._extract_state(info)
        _ = (action, reward)
        self._current.append_state(next_state)
        if terminated or truncated:
            self.end_episode()

    def end_episode(self) -> None:
        self._ensure_open()
        if self._current is None:
            raise RuntimeError("No active episode to end.")
        episode = self._current
        self._current = None
        self._pending_episodes.append(episode)

        if self.write_interval > 0 and len(self._pending_episodes) >= self.write_interval:
            self._flush_pending()

    def close(self) -> None:
        if self._closed:
            return
        if self._current is not None:
            self.end_episode()
        if self._pending_episodes:
            self._flush_pending()

        self._file.attrs["total_episodes"] = self._episode_index
        self._file.flush()
        self._file.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Writer is closed.")

    @staticmethod
    def _extract_state(info: dict[str, Any]) -> AgentState:
        state = info.get("state")
        if not isinstance(state, AgentState):
            raise KeyError("info['state'] missing. Wrap env with StateCaptureWrapper first.")
        return state

    def _flush_pending(self) -> None:
        for episode in self._pending_episodes:
            self._write_episode(episode, self._episode_index)
            self._episode_index += 1
        self._pending_episodes.clear()
        self._file.flush()

    def _write_episode(self, episode: _EpisodeBuffer, episode_index: int) -> None:
        episode_group = self._file.create_group(f"episode_{episode_index}")
        episode_group.attrs["num_states"] = episode.num_states

        utf8 = h5py.string_dtype(encoding="utf-8")

        episode_group.create_dataset("mission", data=np.asarray(episode.missions, dtype=object), dtype=utf8)
        episode_group.create_dataset("agent_pos", data=np.asarray(episode.agent_pos, dtype=np.int32))
        episode_group.create_dataset("agent_dir", data=np.asarray(episode.agent_dir, dtype=np.int8))
        episode_group.create_dataset(
            "grid",
            data=np.asarray(episode.grid, dtype=np.uint8),
            compression=self.compression,
        )
        episode_group.create_dataset("carrying", data=np.asarray(episode.carrying, dtype=object), dtype=utf8)
