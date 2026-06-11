from dataclasses import dataclass
from typing import Any

import numpy as np

from ice_offline.store.state._spec import State, StateIO


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


_HOPPER_DT = 0.008
_FORWARD_REWARD_WEIGHT = 1.0
_CTRL_COST_WEIGHT = 1e-3
_HEALTHY_REWARD = 1.0
_HEALTHY_STATE_MIN = -100.0
_HEALTHY_STATE_MAX = 100.0
_HEALTHY_Z_MIN = 0.7
_HEALTHY_ANGLE_MIN = -0.2
_HEALTHY_ANGLE_MAX = 0.2


class HopperConverter:
    def convert_episode(self, trajectory: Any) -> list[HopperState]:
        observations = np.asarray(trajectory.observations, dtype=np.float64)
        actions = np.asarray(trajectory.actions, dtype=np.float64)
        rewards = np.asarray(trajectory.rewards, dtype=np.float64)
        num_states = observations.shape[0]

        qpos0_seq = self._rebuild_rootx(observations, actions, rewards)
        states: list[HopperState] = []
        for state_index in range(num_states):
            obs = observations[state_index]
            qpos = np.asarray(
                [qpos0_seq[state_index], obs[0], obs[1], obs[2], obs[3], obs[4]],
                dtype=np.float64,
            )
            qvel = np.asarray(obs[5:11], dtype=np.float64)
            states.append(HopperState(qpos=qpos, qvel=qvel))
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
            qpos0_seq[state_index] = qpos0_seq[state_index - 1] + x_velocity * _HOPPER_DT
        return qpos0_seq

    def _healthy_reward(self, observation: np.ndarray) -> float:
        z = observation[0]
        angle = observation[1]
        state = observation[1:]
        healthy_state = np.all(
            np.logical_and(_HEALTHY_STATE_MIN < state, state < _HEALTHY_STATE_MAX)
        )
        healthy_z = _HEALTHY_Z_MIN < z
        healthy_angle = _HEALTHY_ANGLE_MIN < angle < _HEALTHY_ANGLE_MAX
        is_healthy = healthy_state and healthy_z and healthy_angle
        return _HEALTHY_REWARD if is_healthy else 0.0
