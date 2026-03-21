
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import h5py
import numpy as np


@dataclass
class _EpisodeBuffer:
    observations: list[str] = field(default_factory=list)

    def append_observation(self, observation: Any) -> None:
        payload = DataTrajectoryManager._to_jsonable(observation)
        self.observations.append(json.dumps(payload, ensure_ascii=True))

    @property
    def num_steps(self) -> int:
        return len(self.observations)

    def to_observation_trajectory(self) -> list[Any]:
        return [json.loads(item) for item in self.observations]


class DataTrajectoryManager:
    """Unified observation trajectory interface for read/write/serialize."""

    def __init__(
        self,
        path: str | Path,
        *,
        mode: str = "r",
        flush_interval: int = 0,
        compression: str | None = "gzip",
    ) -> None:
        if flush_interval < 0:
            raise ValueError("flush_interval must be >= 0.")

        self.path = Path(path)
        self.mode = str(mode).lower()
        self.flush_interval = int(flush_interval)
        self.compression = compression

        self._file = None
        self._episode_keys: list[str] = []
        self._pending_episodes: list[_EpisodeBuffer] = []
        self.episodes: list[list[Any]] = []
        self._episode_index = 0
        self._current: _EpisodeBuffer | None = None
        self._closed = False

        self._open_dataset()

    def num_episodes(self) -> int:
        return len(self._episode_views())

    def episode_length(self, episode_index: int) -> int:
        trajectory = self._episode_at(episode_index)
        return len(trajectory)

    def get_observation(self, episode_index: int, step_index: int) -> Any:
        trajectory = self._episode_at(episode_index)
        num_steps = len(trajectory)
        if step_index < 0 or step_index >= num_steps:
            raise IndexError(
                f"step_index out of range: episode={episode_index}, "
                f"step_index={step_index}, num_steps={num_steps}"
            )
        return trajectory[step_index]

    def iter_episode_observations(self, episode_index: int) -> Iterator[Any]:
        num_steps = self.episode_length(episode_index)
        for step_index in range(num_steps):
            yield self.get_observation(episode_index=episode_index, step_index=step_index)

    def read(self, max_episodes: int | None = None) -> list[list[Any]]:
        if max_episodes is not None and max_episodes <= 0:
            raise ValueError("max_episodes must be > 0 when provided.")
        if self.mode == "r":
            self.episodes = self._load_episodes_from_file(max_episodes=max_episodes)
            return [list(ep) for ep in self.episodes]

        source = self._episode_views()
        total = len(source)
        limit = total if max_episodes is None else min(int(max_episodes), total)
        trajectories: list[list[Any]] = []
        for episode_index in range(limit):
            trajectories.append(source[episode_index])
        return trajectories

    def push_observation(self, observation: Any) -> None:
        self._require_mode("w")
        if self._current is None:
            self._current = _EpisodeBuffer()
        self._current.append_observation(observation)

    def end_episode(self) -> None:
        self._require_mode("w")
        if self._current is None:
            raise RuntimeError("No active episode to end.")
        if self._current.num_steps == 0:
            self._current = None
            return
        self._pending_episodes.append(self._current)
        self.episodes.append(self._current.to_observation_trajectory())
        self._current = None
        self._auto_flush_if_needed()

    def push_episode(self, observations: Iterable[Any]) -> None:
        self._require_mode("w")
        self._current = _EpisodeBuffer()
        for observation in observations:
            self._current.append_observation(observation)
        if self._current.num_steps == 0:
            self._current = None
            return
        self._pending_episodes.append(self._current)
        self.episodes.append(self._current.to_observation_trajectory())
        self._current = None
        self._auto_flush_if_needed()

    def push_episodes(self, episodes: Iterable[Iterable[Any]]) -> None:
        for observations in episodes:
            self.push_episode(observations)

    def flush(self) -> None:
        self._require_mode("w")
        self._flush_pending()

    def serialize_episode(
        self,
        episode_index: int,
        *,
        include_payload: bool = True,
        include_signature: bool = True,
    ) -> dict[str, Any]:
        observations = list(self.iter_episode_observations(episode_index))
        return self.serialize_observations(
            observations,
            include_payload=include_payload,
            include_signature=include_signature,
        )

    @staticmethod
    def serialize_observations(
        observations: Any,
        *,
        include_payload: bool = True,
        include_signature: bool = True,
    ) -> dict[str, Any]:
        payloads = DataTrajectoryManager._observation_payloads(observations)
        result: dict[str, Any] = {"length": len(payloads)}
        if include_payload:
            result["payload"] = payloads
        if include_signature:
            result["signature"] = DataTrajectoryManager._digest_obj(payloads)
        return result

    def close(self) -> None:
        if self._closed:
            return
        if self.mode == "w" and self._file is not None:
            self._current = None
            self._pending_episodes.clear()
            self._file.attrs["total_episodes"] = self._episode_index
            self._file.flush()
        if self._file is not None:
            self._file.close()
        self._closed = True

    def __enter__(self) -> "DataTrajectoryManager":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        _ = (exc_type, exc, tb)
        self.close()

    def _require_mode(self, expected_mode: str):
        if self._closed:
            raise RuntimeError("Manager is closed.")
        if self.mode != expected_mode:
            operation = "read" if expected_mode == "r" else "write"
            raise RuntimeError(f"{operation} operation requires mode='{expected_mode}'.")
        if self._file is None:
            raise RuntimeError("Dataset file is not open.")
        return self._file

    def _open_dataset(self) -> None:
        if self.mode == "w":
            self._open_for_write()
            return
        if self.mode == "r":
            self._open_for_read()
            return
        raise ValueError("mode must be 'r' or 'w'.")

    def _open_for_write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        self._file.attrs["format"] = "observation_trajectory_v1"
        self._file.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        self._file.attrs["flush_interval"] = self.flush_interval

    def _open_for_read(self) -> None:
        self._file = h5py.File(self.path, "r")
        format_name = self._decode_scalar(self._file.attrs.get("format"))
        if format_name != "observation_trajectory_v1":
            self._file.close()
            raise ValueError(f"Unsupported dataset format in '{self.path}'.")
        self._episode_keys = sorted(
            (key for key in self._file.keys() if key.startswith("episode_")),
            key=lambda key: int(key.split("_", maxsplit=1)[1]),
        )
        if not self._episode_keys:
            self._file.close()
            raise ValueError(f"No episode groups found in '{self.path}'.")

    def _auto_flush_if_needed(self) -> None:
        if self.flush_interval > 0 and len(self._pending_episodes) >= self.flush_interval:
            self._flush_pending()

    def _flush_pending(self) -> None:
        file = self._require_mode("w")
        if not self._pending_episodes:
            return
        for episode in self._pending_episodes:
            self._write_episode(file, episode, self._episode_index)
            self._episode_index += 1
        self._pending_episodes.clear()
        file.flush()

    def _write_episode(self, file, episode: _EpisodeBuffer, episode_index: int) -> None:
        episode_group = file.create_group(f"episode_{episode_index}")
        episode_group.attrs["num_steps"] = episode.num_steps

        utf8 = h5py.string_dtype(encoding="utf-8")
        episode_group.create_dataset(
            "observation",
            data=np.asarray(episode.observations, dtype=object),
            dtype=utf8,
            compression=self.compression,
        )

    def _episode_group(self, episode_index: int):
        file = self._require_mode("r")
        if episode_index < 0 or episode_index >= len(self._episode_keys):
            raise IndexError(f"episode_index out of range: {episode_index}")
        key = self._episode_keys[episode_index]
        return file[key]

    @staticmethod
    def _observation_payloads(observations: Any) -> list[Any]:
        if isinstance(observations, list):
            return [DataTrajectoryManager._to_jsonable(item) for item in observations]

        if isinstance(observations, dict):
            lengths = [len(v) for v in observations.values() if hasattr(v, "__len__")]
            if not lengths:
                return [DataTrajectoryManager._to_jsonable(observations)]
            length = min(lengths)
            payloads: list[dict[str, Any]] = []
            for i in range(length):
                step_obs: dict[str, Any] = {}
                for key, value in observations.items():
                    if hasattr(value, "__getitem__"):
                        step_obs[str(key)] = DataTrajectoryManager._to_jsonable(value[i])
                    else:
                        step_obs[str(key)] = DataTrajectoryManager._to_jsonable(value)
                payloads.append(step_obs)
            return payloads

        return [DataTrajectoryManager._to_jsonable(observations)]

    @staticmethod
    def _digest_obj(value: Any) -> str:
        canonical = json.dumps(
            DataTrajectoryManager._to_jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): DataTrajectoryManager._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [DataTrajectoryManager._to_jsonable(v) for v in value]
        return value

    @staticmethod
    def _decode_scalar(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.bytes_):
            return bytes(value).decode("utf-8")
        return str(value)

    def _episode_views(self) -> list[list[Any]]:
        if self.mode == "r":
            if not self.episodes:
                self.episodes = self._load_episodes_from_file(max_episodes=None)
            return [list(ep) for ep in self.episodes]

        trajectories: list[list[Any]] = [list(ep) for ep in self.episodes]
        if self._current is not None and self._current.num_steps > 0:
            trajectories.append(self._current.to_observation_trajectory())
        return trajectories

    def _episode_at(self, episode_index: int) -> list[Any]:
        trajectories = self._episode_views()
        if episode_index < 0 or episode_index >= len(trajectories):
            raise IndexError(f"episode_index out of range: {episode_index}")
        return trajectories[episode_index]

    def _load_episodes_from_file(self, max_episodes: int | None) -> list[list[Any]]:
        file = self._require_mode("r")
        total = len(self._episode_keys)
        limit = total if max_episodes is None else min(int(max_episodes), total)
        episodes: list[list[Any]] = []
        for episode_index in range(limit):
            key = self._episode_keys[episode_index]
            group = file[key]
            raw_num_steps = group.attrs.get("num_steps")
            num_steps = int(raw_num_steps) if raw_num_steps is not None else int(group["observation"].shape[0])
            trajectory: list[Any] = []
            for step_index in range(num_steps):
                raw = self._decode_scalar(group["observation"][step_index])
                trajectory.append(json.loads(raw))
            episodes.append(trajectory)
        return episodes
