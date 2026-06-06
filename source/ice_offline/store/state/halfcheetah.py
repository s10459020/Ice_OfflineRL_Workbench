from dataclasses import dataclass
from typing import Any

import numpy as np

from ice_offline.store.state._spec import State, StateIO


@dataclass(frozen=True)
class HalfCheetahState(State):
    qpos: np.ndarray
    qvel: np.ndarray

    def serialize(self) -> dict[str, Any]:
        return {
            "qpos": np.asarray(self.qpos, dtype=np.float64),
            "qvel": np.asarray(self.qvel, dtype=np.float64),
        }

    @classmethod
    def from_serialized(cls, payload: dict[str, Any]) -> "HalfCheetahState":
        return cls(
            qpos=np.asarray(payload["qpos"], dtype=np.float64),
            qvel=np.asarray(payload["qvel"], dtype=np.float64),
        )


class HalfCheetahStateIO(StateIO):
    def __init__(self, env: Any) -> None:
        self._env = env

    def get_state(self) -> HalfCheetahState:
        base = self._env.unwrapped
        return HalfCheetahState(
            qpos=np.asarray(base.data.qpos, dtype=np.float64).copy(),
            qvel=np.asarray(base.data.qvel, dtype=np.float64).copy(),
        )

    def set_state(self, state: HalfCheetahState) -> None:
        base = self._env.unwrapped
        qpos = np.asarray(state.qpos, dtype=np.float64)
        qvel = np.asarray(state.qvel, dtype=np.float64)
        base.set_state(qpos, qvel)


_HALFCHEETAH_DT = 0.05
_FORWARD_REWARD_WEIGHT = 1.0
_CTRL_COST_WEIGHT = 0.1


class HalfCheetahConverter:
    def convert_episode(self, trajectory: Any) -> list[HalfCheetahState]:
        observations = np.asarray(trajectory.observations, dtype=np.float64)
        actions = np.asarray(trajectory.actions, dtype=np.float64)
        rewards = np.asarray(trajectory.rewards, dtype=np.float64)
        num_states = observations.shape[0]

        qpos0_seq = self._rebuild_rootx(observations, actions, rewards)
        states: list[HalfCheetahState] = []
        for state_index in range(num_states):
            obs = observations[state_index]
            qpos = np.concatenate(
                [
                    np.asarray([qpos0_seq[state_index]], dtype=np.float64),
                    np.asarray(obs[:8], dtype=np.float64),
                ],
                axis=0,
            )
            qvel = np.asarray(obs[8:17], dtype=np.float64)
            states.append(HalfCheetahState(qpos=qpos, qvel=qvel))
        return states

    def _rebuild_rootx(
        self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray
    ) -> np.ndarray:
        num_states = observations.shape[0]
        qpos0_seq = np.zeros(num_states, dtype=np.float64)
        for state_index in range(1, num_states):
            reward_prev = rewards[state_index - 1]
            action_prev = actions[state_index - 1]

            ctrl_cost = _CTRL_COST_WEIGHT * float(np.sum(np.square(action_prev)))
            x_velocity = (reward_prev + ctrl_cost) / _FORWARD_REWARD_WEIGHT
            qpos0_seq[state_index] = qpos0_seq[state_index - 1] + x_velocity * _HALFCHEETAH_DT
        return qpos0_seq
