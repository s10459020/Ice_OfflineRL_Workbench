from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    h5py = None
    _H5PY_IMPORT_ERROR = exc
else:
    _H5PY_IMPORT_ERROR = None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


@dataclass
class _EpisodeBuffer:
    observations: list[str] = field(default_factory=list)

    def append_observation(self, observation: Any) -> None:
        payload = _to_jsonable(observation)
        self.observations.append(json.dumps(payload, ensure_ascii=True))

    @property
    def num_steps(self) -> int:
        return len(self.observations)


class ObservationTrajectoryWriter:
    """Write completed observation trajectories to HDF5."""

    def __init__(
        self,
        output_path: str | Path,
        flush_interval: int = 0,
        compression: str | None = "gzip",
    ) -> None:
        if flush_interval < 0:
            raise ValueError("flush_interval must be >= 0.")
        if h5py is None:  # pragma: no cover
            raise ImportError("h5py is required for ObservationTrajectoryWriter.") from _H5PY_IMPORT_ERROR

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_interval = int(flush_interval)
        self.compression = compression

        self._file = h5py.File(self.output_path, "w")
        self._file.attrs["format"] = "observation_trajectory_v1"
        self._file.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        self._file.attrs["flush_interval"] = self.flush_interval

        self._pending_episodes: list[_EpisodeBuffer] = []
        self._episode_index = 0
        self._current: _EpisodeBuffer | None = None
        self._closed = False

    def push_observation(self, observation: Any) -> None:
        self._ensure_open()
        if self._current is None:
            self._current = _EpisodeBuffer()
        self._current.append_observation(observation)

    def end_episode(self) -> None:
        self._ensure_open()
        if self._current is None:
            raise RuntimeError("No active episode to end.")
        if self._current.num_steps == 0:
            self._current = None
            return
        self._pending_episodes.append(self._current)
        self._current = None
        self._auto_flush_if_needed()

    def push_episode(self, observations: Iterable[Any]) -> None:
        self._ensure_open()
        self._current = _EpisodeBuffer()
        for observation in observations:
            self._current.append_observation(observation)
        if self._current.num_steps == 0:
            self._current = None
            return
        self._pending_episodes.append(self._current)
        self._current = None
        self._auto_flush_if_needed()

    def push_episodes(self, episodes: Iterable[Iterable[Any]]) -> None:
        for observations in episodes:
            self.push_episode(observations)

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
        episode_group.attrs["num_steps"] = episode.num_steps

        utf8 = h5py.string_dtype(encoding="utf-8")
        episode_group.create_dataset(
            "observation",
            data=np.asarray(episode.observations, dtype=object),
            dtype=utf8,
            compression=self.compression,
        )
