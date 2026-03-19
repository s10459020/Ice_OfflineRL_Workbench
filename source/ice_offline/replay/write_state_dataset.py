from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ice_offline.tools.types import State

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

    def append_state(self, state: State) -> None:
        self.missions.append(str(state.mission))
        self.agent_pos.append((int(state.agent_pos[0]), int(state.agent_pos[1])))
        self.agent_dir.append(int(state.agent_dir))
        self.grid.append(np.asarray(state.grid, dtype=np.uint8).copy())
        self.carrying.append(json.dumps(state.carrying, ensure_ascii=True))

    @property
    def num_states(self) -> int:
        return len(self.missions)


class StateDatasetWriter:
    """Write completed state trajectories to HDF5."""

    def __init__(
        self,
        output_path: str | Path,
        flush_interval: int = 0,
        compression: str | None = "gzip",
    ) -> None:
        if flush_interval < 0:
            raise ValueError("flush_interval must be >= 0.")
        if h5py is None:  # pragma: no cover
            raise ImportError("h5py is required for StateDatasetWriter.") from _H5PY_IMPORT_ERROR

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_interval = int(flush_interval)
        self.compression = compression

        self._file = h5py.File(self.output_path, "w")
        self._file.attrs["format"] = "state_dataset_v1"
        self._file.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        self._file.attrs["flush_interval"] = self.flush_interval

        self._pending_episodes: list[_EpisodeBuffer] = []
        self._episode_index = 0
        self._current: _EpisodeBuffer | None = None
        self._closed = False

    def push_state(self, state: State) -> None:
        self._ensure_open()
        if self._current is None:
            self._current = _EpisodeBuffer()
        self._current.append_state(state)

    def end_episode(self) -> None:
        self._ensure_open()
        if self._current is None:
            raise RuntimeError("No active episode to end.")
        if self._current.num_states == 0:
            self._current = None
            return
        self._pending_episodes.append(self._current)
        self._current = None
        self._auto_flush_if_needed()

    def push_episode(self, states: Iterable[State]) -> None:
        self._ensure_open()
        self._current = _EpisodeBuffer()
        for state in states:
            self._current.append_state(state)
        if self._current.num_states == 0:
            self._current = None
            return
        self._pending_episodes.append(self._current)
        self._current = None
        self._auto_flush_if_needed()

    def push_episodes(self, episodes: Iterable[Iterable[State]]) -> None:
        for states in episodes:
            self.push_episode(states)

    def flush(self) -> None:
        self._ensure_open()
        self._flush_pending()

    def close(self) -> None:
        if self._closed:
            return
        # Drop unfinished and unflushed data by design.
        self._current = None
        self._pending_episodes.clear()

        self._file.attrs["total_episodes"] = self._episode_index
        self._file.flush()
        self._file.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Writer is closed.")

    def _auto_flush_if_needed(self) -> None:
        if self.flush_interval > 0 and len(self._pending_episodes) >= self.flush_interval:
            self._flush_pending()

    def _flush_pending(self) -> None:
        if not self._pending_episodes:
            return
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
