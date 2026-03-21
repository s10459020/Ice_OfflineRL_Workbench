from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import h5py
import numpy as np

from ice_offline.tools.types import State


class StateDataset:
    """State dataset with in-memory storage + HDF5 read/write + signature helpers."""

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

        self.episodes: list[list[State]] = []
        self.flush_pointer = 0
        self._episode_open = False
        self._closed = False

        self._file: h5py.File | None = None
        self._episode_keys: list[str] = []

        if self.mode == "w":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = h5py.File(self.path, "w")
            self._file.attrs["format"] = "state_dataset_v1"
            self._file.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            self._file.attrs["flush_interval"] = self.flush_interval
            return

        if self.mode == "r":
            self._file = h5py.File(self.path, "r")
            format_name = self._decode_scalar(self._file.attrs.get("format"))
            if format_name != "state_dataset_v1":
                self._file.close()
                raise ValueError(f"Unsupported dataset format in '{self.path}'.")

            self._episode_keys = sorted(
                (key for key in self._file.keys() if key.startswith("episode_")),
                key=lambda key: int(key.split("_", maxsplit=1)[1]),
            )
            if not self._episode_keys:
                self._file.close()
                raise ValueError(f"No episode groups found in '{self.path}'.")
            self.read()
            return

        raise ValueError("mode must be 'r' or 'w'.")

    ###############################################################################
    # Query API
    ###############################################################################
    def num_episodes(self) -> int:
        return len(self.episodes)

    def episode_length(self, episode_index: int) -> int:
        return len(self.episodes[self._require_episode_index(episode_index)])

    def get_state(self, episode_index: int, state_index: int) -> State:
        sequence = self.episodes[self._require_episode_index(episode_index)]
        index = self._require_state_index(sequence, state_index, episode_index)
        return sequence[index]

    def iter_episode_states(self, episode_index: int) -> Iterator[State]:
        for state in self.episodes[self._require_episode_index(episode_index)]:
            yield state

    def _require_episode_index(self, episode_index: int) -> int:
        if episode_index < 0 or episode_index >= len(self.episodes):
            raise IndexError(f"episode_index out of range: {episode_index}")
        return episode_index

    @staticmethod
    def _require_state_index(sequence: list[State], state_index: int, episode_index: int) -> int:
        if state_index < 0 or state_index >= len(sequence):
            raise IndexError(
                f"state_index out of range: episode={episode_index}, "
                f"state_index={state_index}, num_states={len(sequence)}"
            )
        return state_index

    ###############################################################################
    # Data Operations
    ###############################################################################
    def push_state(self, state: State) -> None:
        self._ensure_open()
        if not self._episode_open:
            self.episodes.append([])
            self._episode_open = True
        self.episodes[-1].append(state)

    def end_episode(self) -> None:
        self._ensure_open()
        if not self._episode_open:
            raise RuntimeError("No active episode to end.")
        self._episode_open = False
        if self.mode == "w":
            self._auto_flush_if_needed()

    def push_episode(self, states: Iterable[State]) -> None:
        self._ensure_open()
        sequence = list(states)
        if not sequence:
            return
        self.episodes.append(sequence)
        if self.mode == "w":
            self._auto_flush_if_needed()

    def push_episodes(self, episodes: Iterable[Iterable[State]]) -> None:
        for states in episodes:
            self.push_episode(states)

    def _auto_flush_if_needed(self) -> None:
        pending_count = self._flush_limit() - self.flush_pointer
        if self.flush_interval > 0 and pending_count >= self.flush_interval:
            self._flush_pending()

    def _flush_limit(self) -> int:
        if self._episode_open:
            return max(0, len(self.episodes) - 1)
        return len(self.episodes)

    ###############################################################################
    # Read API
    ###############################################################################
    def read(self, max_episodes: int | None = None) -> list[list[State]]:
        self._require_mode("r")
        if max_episodes is not None and max_episodes <= 0:
            raise ValueError("max_episodes must be > 0 when provided.")
        self.episodes = self._load_episodes_from_file(max_episodes=max_episodes)
        return [list(seq) for seq in self.episodes]

    def _load_episodes_from_file(self, max_episodes: int | None) -> list[list[State]]:
        file = self._require_mode("r")
        total = len(self._episode_keys)
        limit = total if max_episodes is None else min(int(max_episodes), total)
        episodes: list[list[State]] = []
        for episode_index in range(limit):
            group = file[self._episode_keys[episode_index]]
            raw_num_states = group.attrs.get("num_states")
            num_states = int(raw_num_states) if raw_num_states is not None else int(group["grid"].shape[0])
            states: list[State] = []
            for state_index in range(num_states):
                states.append(self._decode_state(group, state_index))
            episodes.append(states)
        return episodes

    def _decode_state(self, group: Any, state_index: int) -> State:
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

    @staticmethod
    def _decode_scalar(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.bytes_):
            return bytes(value).decode("utf-8")
        return str(value)

    ###############################################################################
    # Write API
    ###############################################################################
    def write(self) -> None:
        self._require_mode("w")
        self._flush_pending()

    def flush(self) -> None:
        self.write()

    def _flush_pending(self) -> None:
        file = self._require_mode("w")
        flush_limit = self._flush_limit()
        if self.flush_pointer >= flush_limit:
            return
        for episode_index in range(self.flush_pointer, flush_limit):
            self._write_episode(file, self.episodes[episode_index], episode_index)
        self.flush_pointer = flush_limit
        file.flush()

    def _write_episode(self, file, state_sequence: list[State], episode_index: int) -> None:
        episode_group = file.create_group(f"episode_{episode_index}")
        episode_group.attrs["num_states"] = len(state_sequence)

        utf8 = h5py.string_dtype(encoding="utf-8")
        missions, agent_pos, agent_dir, grid, carrying = self._encode_state_sequence(state_sequence)

        episode_group.create_dataset("mission", data=np.asarray(missions, dtype=object), dtype=utf8)
        episode_group.create_dataset("agent_pos", data=np.asarray(agent_pos, dtype=np.int32))
        episode_group.create_dataset("agent_dir", data=np.asarray(agent_dir, dtype=np.int8))
        episode_group.create_dataset("grid", data=np.asarray(grid, dtype=np.uint8), compression=self.compression)
        episode_group.create_dataset("carrying", data=np.asarray(carrying, dtype=object), dtype=utf8)

    def _encode_state_sequence(
        self, state_sequence: list[State]
    ) -> tuple[list[str], list[tuple[int, int]], list[int], list[np.ndarray], list[str]]:
        missions = [str(s.mission) for s in state_sequence]
        agent_pos = [(int(s.agent_pos[0]), int(s.agent_pos[1])) for s in state_sequence]
        agent_dir = [int(s.agent_dir) for s in state_sequence]
        grid = [np.asarray(s.grid, dtype=np.uint8).copy() for s in state_sequence]
        carrying = [json.dumps(s.carrying, ensure_ascii=True) for s in state_sequence]
        return missions, agent_pos, agent_dir, grid, carrying

    ###############################################################################
    # Serialize API
    ###############################################################################
    def serialize_state(self, episode_index: int, state_index: int) -> str:
        return self.serialize_states([self.get_state(episode_index, state_index)])

    def serialize_episode(self, episode_index: int) -> str:
        return self.serialize_states(self.episodes[self._require_episode_index(episode_index)])

    def serialize_trajectory(self) -> str:
        all_states: list[State] = []
        for episode in self.episodes:
            all_states.extend(episode)
        return self.serialize_states(all_states)

    @staticmethod
    def serialize_states(states: list[Any]) -> str:
        payloads = []
        for state in states:
            grid = np.asarray(state.grid, dtype=np.uint8)
            payloads.append(
                {
                    "mission": str(state.mission),
                    "agent_pos": [int(state.agent_pos[0]), int(state.agent_pos[1])],
                    "agent_dir": int(state.agent_dir),
                    "carrying": json.loads(json.dumps(state.carrying, ensure_ascii=True)),
                    "grid_signature": hashlib.sha256(grid.tobytes()).hexdigest()[:16],
                }
            )
        canonical = json.dumps(payloads, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    ###############################################################################
    # Lifecycle API
    ###############################################################################
    def close(self) -> None:
        if self._closed:
            return
        if self.mode == "w" and self._file is not None:
            if self._episode_open and self.episodes:
                self.episodes.pop()
            self._episode_open = False
            self._file.attrs["total_episodes"] = self.flush_pointer
            self._file.flush()
        if self._file is not None:
            self._file.close()
        self._closed = True

    def __enter__(self) -> "StateDataset":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        _ = (exc_type, exc, tb)
        self.close()

    ###############################################################################
    # Internal Helpers
    ###############################################################################
    def _require_mode(self, expected_mode: str):
        self._ensure_open()
        if self.mode != expected_mode:
            operation = "read" if expected_mode == "r" else "write"
            raise RuntimeError(f"{operation} operation requires mode='{expected_mode}'.")
        if self._file is None:
            raise RuntimeError("Dataset file is not open.")
        return self._file

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Dataset is closed.")
