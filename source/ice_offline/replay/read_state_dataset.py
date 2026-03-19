from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ice_offline.tools.types import State

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    h5py = None
    _H5PY_IMPORT_ERROR = exc
else:
    _H5PY_IMPORT_ERROR = None


class StateDatasetReader:
    """Read MiniGrid state trajectories written by StateDatasetWriter."""

    def __init__(self, input_path: str | Path) -> None:
        if h5py is None:  # pragma: no cover
            raise ImportError("h5py is required for StateDatasetReader.") from _H5PY_IMPORT_ERROR

        self.input_path = Path(input_path)
        self._file = h5py.File(self.input_path, "r")
        self._closed = False

        format_name = self._file.attrs.get("format")
        if self._decode_scalar(format_name) != "state_dataset_v1":
            self._file.close()
            raise ValueError(f"Unsupported dataset format in '{self.input_path}'.")

        self._episode_keys = sorted(
            (key for key in self._file.keys() if key.startswith("episode_")),
            key=lambda key: int(key.split("_", maxsplit=1)[1]),
        )
        if not self._episode_keys:
            self._file.close()
            raise ValueError(f"No episode groups found in '{self.input_path}'.")

    @property
    def num_episodes(self) -> int:
        return len(self._episode_keys)

    @property
    def total_episodes(self) -> int:
        raw = self._file.attrs.get("total_episodes")
        if raw is None:
            return self.num_episodes
        return int(raw)

    def episode_length(self, episode_index: int) -> int:
        group = self._episode_group(episode_index)
        raw = group.attrs.get("num_states")
        if raw is not None:
            return int(raw)
        return int(group["grid"].shape[0])

    def get_state(self, episode_index: int, state_index: int) -> State:
        group = self._episode_group(episode_index)
        num_states = self.episode_length(episode_index)
        if state_index < 0 or state_index >= num_states:
            raise IndexError(
                f"state_index out of range: episode={episode_index}, "
                f"state_index={state_index}, num_states={num_states}"
            )

        mission = self._decode_scalar(group["mission"][state_index])
        agent_pos_raw = np.asarray(group["agent_pos"][state_index], dtype=np.int32)
        agent_dir = int(group["agent_dir"][state_index])
        grid = np.asarray(group["grid"][state_index], dtype=np.uint8).copy()
        carrying_json = self._decode_scalar(group["carrying"][state_index])
        carrying = json.loads(carrying_json)

        return State(
            mission=mission,
            agent_pos=(int(agent_pos_raw[0]), int(agent_pos_raw[1])),
            agent_dir=agent_dir,
            grid=grid,
            carrying=carrying,
        )

    def iter_episode_states(self, episode_index: int) -> Iterator[State]:
        num_states = self.episode_length(episode_index)
        for state_index in range(num_states):
            yield self.get_state(episode_index=episode_index, state_index=state_index)

    def read(self, max_episodes: int | None = None) -> list[list[State]]:
        if max_episodes is not None and max_episodes <= 0:
            raise ValueError("max_episodes must be > 0 when provided.")
        limit = self.num_episodes if max_episodes is None else min(int(max_episodes), self.num_episodes)
        trajectories: list[list[State]] = []
        for episode_index in range(limit):
            trajectories.append(list(self.iter_episode_states(episode_index)))
        return trajectories

    def close(self) -> None:
        if self._closed:
            return
        self._file.close()
        self._closed = True

    def __enter__(self) -> StateDatasetReader:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        _ = (exc_type, exc, tb)
        self.close()

    def _episode_group(self, episode_index: int):
        self._ensure_open()
        if episode_index < 0 or episode_index >= self.num_episodes:
            raise IndexError(f"episode_index out of range: {episode_index}")
        key = self._episode_keys[episode_index]
        return self._file[key]

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Reader is closed.")

    @staticmethod
    def _decode_scalar(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.bytes_):
            return bytes(value).decode("utf-8")
        return str(value)
