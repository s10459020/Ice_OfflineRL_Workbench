from __future__ import annotations

import io
from typing import Any

import numpy as np

from .state_types import AgentState

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    Image = None
    _PIL_IMPORT_ERROR = exc
else:
    _PIL_IMPORT_ERROR = None

try:
    from minigrid.core.constants import OBJECT_TO_IDX
except ImportError:  # pragma: no cover
    OBJECT_TO_IDX = {
        "unseen": 0,
        "empty": 1,
        "wall": 2,
        "floor": 3,
        "door": 4,
        "key": 5,
        "ball": 6,
        "box": 7,
        "goal": 8,
        "lava": 9,
        "agent": 10,
    }
_AGENT_OBJECT_IDX = int(OBJECT_TO_IDX["agent"])


def convert_fullobs(episode: Any) -> list[AgentState]:
    observations = _read_observations(episode)
    for key in ("image", "direction", "mission"):
        if key not in observations:
            raise KeyError(f"Missing observations['{key}'] in Minari episode.")

    image_seq = observations["image"]
    dir_seq = observations["direction"]
    mission_seq = observations["mission"]
    num_states = len(dir_seq)
    if len(image_seq) != num_states or len(mission_seq) != num_states:
        raise ValueError("Inconsistent trajectory lengths among image/direction/mission.")

    states: list[AgentState] = []
    for index in range(num_states):
        grid = _coerce_grid(image_seq[index])
        states.append(
            AgentState(
                mission=_decode_text(mission_seq[index]),
                agent_pos=_find_agent_pos(grid, state_index=index),
                agent_dir=int(dir_seq[index]),
                grid=grid,
                carrying=None,
            )
        )
    return states


def _read_observations(episode: Any) -> dict[str, Any]:
    # Minari episode objects expose trajectory fields via `.observations`.
    if hasattr(episode, "observations"):
        observations = episode.observations
    else:
        observations = episode
    if not isinstance(observations, dict):
        raise TypeError("Episode observations must be dict-like for fullobs conversion.")
    return observations


def _coerce_grid(image_value: Any) -> np.ndarray:
    arr = np.asarray(image_value)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return np.asarray(arr, dtype=np.uint8).copy()

    if arr.ndim == 1 and _looks_like_jpeg(arr):
        if Image is None:  # pragma: no cover
            raise ImportError("Pillow is required to decode JPEG-packed observations.") from _PIL_IMPORT_ERROR
        jpeg_bytes = bytes(np.asarray(arr, dtype=np.uint8).tolist())
        decoded = np.asarray(Image.open(io.BytesIO(jpeg_bytes)), dtype=np.uint8)
        if decoded.ndim == 3 and decoded.shape[-1] == 3:
            return decoded.copy()
        raise ValueError(f"Decoded JPEG observation has invalid shape={decoded.shape}.")

    if arr.ndim == 1 and arr.size % 3 == 0:
        pixels = arr.size // 3
        side = int(np.sqrt(pixels))
        if side * side * 3 == arr.size:
            return np.asarray(arr.reshape(side, side, 3), dtype=np.uint8).copy()

    if arr.ndim == 2 and arr.shape[-1] == 3:
        pixels = arr.shape[0]
        side = int(np.sqrt(pixels))
        if side * side == pixels:
            return np.asarray(arr.reshape(side, side, 3), dtype=np.uint8).copy()

    raise ValueError(f"Unsupported fullobs image shape={arr.shape}, dtype={arr.dtype}.")


def _looks_like_jpeg(arr: np.ndarray) -> bool:
    if arr.ndim != 1 or arr.size < 4:
        return False
    a = np.asarray(arr, dtype=np.uint8)
    return int(a[0]) == 0xFF and int(a[1]) == 0xD8


def _find_agent_pos(grid: np.ndarray, state_index: int) -> tuple[int, int]:
    agent_coords = np.argwhere(np.asarray(grid[:, :, 0], dtype=np.int16) == _AGENT_OBJECT_IDX)
    if len(agent_coords) != 1:
        raise ValueError(
            f"Expected exactly one agent cell in grid[:, :, 0] at state_index={state_index}, found={len(agent_coords)}."
        )
    x, y = agent_coords[0]
    return int(x), int(y)


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


convert_fullobs_episode = convert_fullobs
