
import hashlib
import json
from typing import Any

import numpy as np


def serialize_state_trajectory(
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
        result["signature"] = _digest_obj(payloads)
    return result


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


def _digest_obj(value: Any) -> str:
    canonical = json.dumps(_to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


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

