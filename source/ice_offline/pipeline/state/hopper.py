from dataclasses import dataclass
from typing import Any

import numpy as np

from ice_offline.pipeline.state._spec import State, StateIO


@dataclass(frozen=True)
class HopperState(State):
    qpos: np.ndarray
    qvel: np.ndarray

    def serialize(self) -> dict[str, Any]:
        return {
            "qpos": np.asarray(self.qpos, dtype=np.float64),
            "qvel": np.asarray(self.qvel, dtype=np.float64),
        }

    @classmethod
    def from_serialized(cls, payload: dict[str, Any]) -> "HopperState":
        return cls(
            qpos=np.asarray(payload["qpos"], dtype=np.float64),
            qvel=np.asarray(payload["qvel"], dtype=np.float64),
        )


class HopperStateIO(StateIO):
    def __init__(self, env: Any) -> None:
        self._env = env

    def get_state(self) -> HopperState:
        base = self._env.unwrapped
        return HopperState(
            qpos=np.asarray(base.data.qpos, dtype=np.float64).copy(),
            qvel=np.asarray(base.data.qvel, dtype=np.float64).copy(),
        )

    def set_state(self, state: HopperState) -> None:
        base = self._env.unwrapped
        qpos = np.asarray(state.qpos, dtype=np.float64)
        qvel = np.asarray(state.qvel, dtype=np.float64)
        base.set_state(qpos, qvel)
