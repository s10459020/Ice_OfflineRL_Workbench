from dataclasses import dataclass
from typing import Any

import numpy as np

from ice_offline.store.state._spec import State, StateIO


@dataclass(frozen=True)
class Walker2dState(State):
    qpos: np.ndarray
    qvel: np.ndarray

    def serialize(self) -> dict[str, Any]:
        return {
            "qpos": np.asarray(self.qpos, dtype=np.float64),
            "qvel": np.asarray(self.qvel, dtype=np.float64),
        }

    @classmethod
    def from_serialized(cls, payload: dict[str, Any]) -> "Walker2dState":
        return cls(
            qpos=np.asarray(payload["qpos"], dtype=np.float64),
            qvel=np.asarray(payload["qvel"], dtype=np.float64),
        )


class Walker2dStateIO(StateIO):
    def __init__(self, env: Any) -> None:
        self._env = env

    def get_state(self) -> Walker2dState:
        base = self._env.unwrapped
        return Walker2dState(
            qpos=np.asarray(base.data.qpos, dtype=np.float64).copy(),
            qvel=np.asarray(base.data.qvel, dtype=np.float64).copy(),
        )

    def set_state(self, state: Walker2dState) -> None:
        base = self._env.unwrapped
        qpos = np.asarray(state.qpos, dtype=np.float64)
        qvel = np.asarray(state.qvel, dtype=np.float64)
        base.set_state(qpos, qvel)


_WALKER2D_DT = 0.008
_FORWARD_REWARD_WEIGHT = 1.0
_CTRL_COST_WEIGHT = 1e-3
_HEALTHY_REWARD = 1.0
_HEALTHY_Z_MIN = 0.8
_HEALTHY_Z_MAX = 2.0
_HEALTHY_ANGLE_MIN = -1.0
_HEALTHY_ANGLE_MAX = 1.0


class Walker2dConverter:
    def convert_episode(self, trajectory: Any) -> list[Walker2dState]:
        observations = np.asarray(trajectory.observations, dtype=np.float64)
        actions = np.asarray(trajectory.actions, dtype=np.float64)
        rewards = np.asarray(trajectory.rewards, dtype=np.float64)
        num_states = observations.shape[0]

        qpos0_seq = self._rebuild_rootx(observations, actions, rewards)
        states: list[Walker2dState] = []
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
            states.append(Walker2dState(qpos=qpos, qvel=qvel))
        return states

    def _rebuild_rootx(
        self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray
    ) -> np.ndarray:
        num_states = observations.shape[0]
        qpos0_seq = np.zeros(num_states, dtype=np.float64)
        for state_index in range(1, num_states):
            reward_prev = rewards[state_index - 1]
            action_prev = actions[state_index - 1]
            obs_curr = observations[state_index]

            healthy_reward = self._healthy_reward(obs_curr)
            ctrl_cost = _CTRL_COST_WEIGHT * float(np.sum(np.square(action_prev)))
            x_velocity = (reward_prev + ctrl_cost - healthy_reward) / _FORWARD_REWARD_WEIGHT
            qpos0_seq[state_index] = qpos0_seq[state_index - 1] + x_velocity * _WALKER2D_DT
        return qpos0_seq

    def _healthy_reward(self, observation: np.ndarray) -> float:
        z = observation[0]
        angle = observation[1]
        healthy_z = _HEALTHY_Z_MIN < z < _HEALTHY_Z_MAX
        healthy_angle = _HEALTHY_ANGLE_MIN < angle < _HEALTHY_ANGLE_MAX
        is_healthy = healthy_z and healthy_angle
        return _HEALTHY_REWARD if is_healthy else 0.0
