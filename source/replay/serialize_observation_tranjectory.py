from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


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
        result["signature"] = _digest_obj(payloads)
    return result


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

