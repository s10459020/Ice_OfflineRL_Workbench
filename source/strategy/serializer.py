from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def digest_obj(value: Any) -> str:
    canonical = json.dumps(_to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def serialize_state_tranjectory(
    states: list[Any],
    *,
    include_payload: bool = True,
    include_signature: bool = True,
    ignore_carrying: bool = False,
    normalize_agent_cell: bool = False,
) -> dict[str, Any]:
    payloads = [
        _state_payload(s, ignore_carrying=ignore_carrying, normalize_agent_cell=normalize_agent_cell) for s in states
    ]
    result: dict[str, Any] = {"length": len(payloads)}
    if include_payload:
        result["payload"] = payloads
    if include_signature:
        result["signature"] = digest_obj(payloads)
    return result


def serialize_observation_tranjectory(
    observations: Any,
    *,
    include_payload: bool = True,
    include_signature: bool = True,
) -> dict[str, Any]:
    payloads = _observation_payloads(observations)
    result: dict[str, Any] = {"length": len(payloads)}
    if include_payload:
        result["payload"] = payloads
    if include_signature:
        result["signature"] = digest_obj(payloads)
    return result


def state_trajectory_signature(states: list[Any]) -> str:
    return serialize_state_tranjectory(states, include_payload=False, include_signature=True)["signature"]


def _state_payload(
    state: Any,
    *,
    ignore_carrying: bool = False,
    normalize_agent_cell: bool = False,
) -> dict[str, Any]:
    grid = np.asarray(state.grid, dtype=np.uint8)
    if normalize_agent_cell:
        x, y = int(state.agent_pos[0]), int(state.agent_pos[1])
        if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
            grid = grid.copy()
            grid[x, y, :] = np.array([1, 0, 0], dtype=np.uint8)
    return {
        "mission": str(state.mission),
        "agent_pos": [int(state.agent_pos[0]), int(state.agent_pos[1])],
        "agent_dir": int(state.agent_dir),
        "carrying": None if ignore_carrying else state.carrying,
        "grid_shape": list(grid.shape),
        "grid_digest": hashlib.sha256(grid.tobytes()).hexdigest()[:16],
    }


def _observation_payloads(observations: Any) -> list[Any]:
    if isinstance(observations, list):
        return [_to_jsonable(item) for item in observations]

    if isinstance(observations, dict):
        lengths = [len(v) for v in observations.values() if hasattr(v, "__len__")]
        if not lengths:
            return [_to_jsonable(observations)]
        length = min(lengths)
        payloads: list[dict[str, Any]] = []
        for i in range(length):
            step_obs: dict[str, Any] = {}
            for key, value in observations.items():
                if hasattr(value, "__getitem__"):
                    step_obs[str(key)] = _to_jsonable(value[i])
                else:
                    step_obs[str(key)] = _to_jsonable(value)
            payloads.append(step_obs)
        return payloads

    return [_to_jsonable(observations)]


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
