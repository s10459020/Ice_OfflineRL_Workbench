
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ice_offline.replay.state import State
from ice_offline.replay.state_io_wrapper import ensure_state_io


class StateReplayWrapper(gym.Wrapper):
    """
    Replay recorded (state, observation, action, reward) trajectories.

    reset():
      - selects one episode
      - sets env to the first state in that sequence
      - returns the first recorded observation

    act():
      - returns action from current transition index

    step(action=None):
      - advances by one recorded transition
      - if action is provided, it is checked against recorded action
    """

    def __init__(
        self,
        env: gym.Env,
        states: list[list[State]],
        observations: list[list[dict[str, Any]]],
        actions: list[list[int]],
        rewards: list[list[float]],
        *,
        random_episode: bool = False,
    ) -> None:
        super().__init__(ensure_state_io(env))
        if not states:
            raise ValueError("states must contain at least one episode.")
        if not observations:
            raise ValueError("observations must contain at least one episode.")
        if not actions:
            raise ValueError("actions must contain at least one episode.")
        if not rewards:
            raise ValueError("rewards must contain at least one episode.")
        num_episodes = len(states)
        if len(observations) != num_episodes or len(actions) != num_episodes or len(rewards) != num_episodes:
            raise ValueError("states/observations/actions/rewards must have same episode count.")
        if any(len(ep) == 0 for ep in states):
            raise ValueError("Each state sequence episode must contain at least one state.")
        for episode_index, episode_states in enumerate(states):
            expected_transitions = len(episode_states) - 1
            if len(observations[episode_index]) != len(episode_states):
                raise ValueError(
                    f"Episode {episode_index} observations length mismatch: "
                    f"got {len(observations[episode_index])}, expected {len(episode_states)}."
                )
            if len(actions[episode_index]) != expected_transitions:
                raise ValueError(
                    f"Episode {episode_index} actions length mismatch: "
                    f"got {len(actions[episode_index])}, expected {expected_transitions}."
                )
            if len(rewards[episode_index]) != expected_transitions:
                raise ValueError(
                    f"Episode {episode_index} rewards length mismatch: "
                    f"got {len(rewards[episode_index])}, expected {expected_transitions}."
                )
        self.state_sequences = states
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.random_episode = bool(random_episode)
        self._set_state = self.get_wrapper_attr("set_state")

        first_state = self.state_sequences[0][0]
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=first_state.grid.shape,
                    dtype=np.uint8,
                ),
                "direction": spaces.Discrete(4),
                "mission": spaces.Text(max_length=512),
            }
        )

        self._rng = np.random.default_rng()
        self._current_episode_index: int | None = None
        self._state_index: int | None = None
        self._episode_length: int | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        base_obs, base_info = self.env.reset(seed=seed, options=options)
        _ = base_obs

        episode_index = self._pick_episode_index(options=options)
        self._current_episode_index = episode_index
        self._state_index = 0
        self._episode_length = len(self.state_sequences[episode_index])

        state = self.state_sequences[episode_index][0]
        self._set_state(state)
        obs = self.observations[episode_index][0]

        info = dict(base_info)
        info["state"] = state
        info["episode_index"] = episode_index
        info["state_index"] = 0
        info["state_sequence_length"] = self._episode_length
        info["trajectory_length"] = len(self.actions[episode_index])
        return obs, info

    def act(self) -> int:
        self._ensure_active_episode()
        episode_index = int(self._current_episode_index)
        action_index = int(self._state_index)
        episode_actions = self.actions[episode_index]
        if action_index >= len(episode_actions):
            raise RuntimeError("No transition action available at terminal state. Call reset().")
        return int(episode_actions[action_index])

    def step(self, action: int | None = None):
        self._ensure_active_episode()

        next_state_index = int(self._state_index) + 1
        episode_index = int(self._current_episode_index)
        episode_actions = self.actions[episode_index]
        episode_rewards = self.rewards[episode_index]
        transition_index = int(self._state_index)
        if transition_index >= len(episode_actions):
            state = self.state_sequences[episode_index][int(self._state_index)]
            obs = self.observations[episode_index][int(self._state_index)]
            reward = 0.0
            terminated = True
            truncated = False
            info = {
                "state": state,
                "episode_index": episode_index,
                "state_index": int(self._state_index),
                "state_sequence_length": int(self._episode_length),
                "trajectory_length": len(episode_actions),
                "replay_frozen": True,
            }
            return obs, reward, terminated, truncated, info

        recorded_action = int(episode_actions[transition_index])
        reward = float(episode_rewards[transition_index])
        action_mismatch = action is not None and int(action) != recorded_action

        if next_state_index >= int(self._episode_length):
            state = self.state_sequences[episode_index][int(self._state_index)]
            obs = self.observations[episode_index][int(self._state_index)]
            terminated = True
            truncated = False
            info = {
                "state": state,
                "episode_index": episode_index,
                "state_index": int(self._state_index),
                "state_sequence_length": int(self._episode_length),
                "trajectory_length": len(episode_actions),
                "transition_action": recorded_action,
                "action_mismatch": action_mismatch,
                "replay_frozen": True,
            }
            return obs, reward, terminated, truncated, info

        state = self.state_sequences[episode_index][next_state_index]
        self._state_index = next_state_index
        self._set_state(state)

        obs = self.observations[episode_index][next_state_index]
        terminated = next_state_index == int(self._episode_length) - 1
        truncated = False
        info = {
            "state": state,
            "episode_index": episode_index,
            "state_index": next_state_index,
            "state_sequence_length": int(self._episode_length),
            "trajectory_length": len(episode_actions),
            "transition_action": recorded_action,
            "action_mismatch": action_mismatch,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    def _pick_episode_index(self, options: dict[str, Any] | None) -> int:
        if options and "episode_index" in options:
            episode_index = int(options["episode_index"])
            if episode_index < 0 or episode_index >= len(self.state_sequences):
                raise IndexError(f"episode_index out of range: {episode_index}")
            return episode_index

        if self.random_episode:
            return int(self._rng.integers(low=0, high=len(self.state_sequences)))

        if self._current_episode_index is None:
            return 0
        return (int(self._current_episode_index) + 1) % len(self.state_sequences)

    def _ensure_active_episode(self) -> None:
        if self._current_episode_index is None or self._state_index is None or self._episode_length is None:
            raise RuntimeError("Call reset() before step().")

