
import json
from typing import Any

import gymnasium as gym
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj
import numpy as np

from ice_offline.tools.types import State, Transition


class StateReplayWrapper(gym.Wrapper):
    """
    Replay recorded (state, action, reward, next_state) transitions.

    reset():
      - selects one episode
      - sets env to the first state in that sequence

    act():
      - returns action from current transition index

    step(action=None):
      - advances by one recorded transition
      - if action is provided, it is checked against recorded action
    """

    def __init__(
        self,
        env: gym.Env,
        state_sequences: list[list[State]],
        trajectories: list[list[Transition]],
        *,
        random_episode: bool = False,
    ) -> None:
        super().__init__(env)
        if not state_sequences:
            raise ValueError("state_sequences must contain at least one episode.")
        if not trajectories:
            raise ValueError("trajectories must contain at least one episode.")
        if len(state_sequences) != len(trajectories):
            raise ValueError("state_sequences and trajectories must have same episode count.")
        if any(len(ep) == 0 for ep in state_sequences):
            raise ValueError("Each state sequence episode must contain at least one state.")
        for episode_index, states in enumerate(state_sequences):
            transitions = trajectories[episode_index]
            expected = len(states) - 1
            if len(transitions) != expected:
                raise ValueError(
                    f"Episode {episode_index} transition length mismatch: "
                    f"got {len(transitions)}, expected {expected}."
                )
        self.state_sequences = state_sequences
        self.trajectories = trajectories
        self.random_episode = bool(random_episode)

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
        self._apply_state_to_env(state)
        obs = self._state_to_obs(state)

        info = dict(base_info)
        info["state"] = state
        info["episode_index"] = episode_index
        info["state_index"] = 0
        info["state_sequence_length"] = self._episode_length
        info["trajectory_length"] = len(self.trajectories[episode_index])
        return obs, info

    def act(self) -> int:
        self._ensure_active_episode()
        episode_index = int(self._current_episode_index)
        transition_index = int(self._state_index)
        transitions = self.trajectories[episode_index]
        if transition_index >= len(transitions):
            raise RuntimeError("No transition action available at terminal state. Call reset().")
        return int(transitions[transition_index].action)

    def step(self, action: int | None = None):
        self._ensure_active_episode()

        next_state_index = int(self._state_index) + 1
        episode_index = int(self._current_episode_index)
        transitions = self.trajectories[episode_index]
        transition_index = int(self._state_index)
        if transition_index >= len(transitions):
            state = self.state_sequences[episode_index][int(self._state_index)]
            obs = self._state_to_obs(state)
            reward = 0.0
            terminated = True
            truncated = False
            info = {
                "state": state,
                "episode_index": episode_index,
                "state_index": int(self._state_index),
                "state_sequence_length": int(self._episode_length),
                "trajectory_length": len(transitions),
                "replay_frozen": True,
            }
            return obs, reward, terminated, truncated, info

        transition = transitions[transition_index]
        recorded_action = int(transition.action)
        reward = float(transition.reward)
        action_mismatch = action is not None and int(action) != recorded_action

        if next_state_index >= int(self._episode_length):
            state = self.state_sequences[episode_index][int(self._state_index)]
            obs = self._state_to_obs(state)
            terminated = True
            truncated = False
            info = {
                "state": state,
                "episode_index": episode_index,
                "state_index": int(self._state_index),
                "state_sequence_length": int(self._episode_length),
                "trajectory_length": len(transitions),
                "transition_action": recorded_action,
                "action_mismatch": action_mismatch,
                "replay_frozen": True,
            }
            return obs, reward, terminated, truncated, info

        state = self.state_sequences[episode_index][next_state_index]
        self._state_index = next_state_index
        self._apply_state_to_env(state)

        obs = self._state_to_obs(state)
        terminated = next_state_index == int(self._episode_length) - 1
        truncated = False
        info = {
            "state": state,
            "episode_index": episode_index,
            "state_index": next_state_index,
            "state_sequence_length": int(self._episode_length),
            "trajectory_length": len(transitions),
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

    @staticmethod
    def _state_to_obs(state: State) -> dict[str, Any]:
        return {
            "image": np.asarray(state.grid, dtype=np.uint8).copy(),
            "direction": int(state.agent_dir),
            "mission": str(state.mission),
        }

    def _apply_state_to_env(self, state: State) -> None:
        # Force env internals to recorded values so replay follows the exact dataset state.
        base = self.env.unwrapped
        self._restore_grid(base, state.grid)
        if hasattr(base, "agent_pos"):
            base.agent_pos = tuple(state.agent_pos)
        if hasattr(base, "agent_dir"):
            base.agent_dir = int(state.agent_dir)
        if hasattr(base, "mission"):
            base.mission = str(state.mission)
        if hasattr(base, "carrying"):
            base.carrying = self._decode_carrying(state.carrying)

    @staticmethod
    def _restore_grid(base_env: Any, encoded_grid: np.ndarray) -> None:
        grid_array = np.asarray(encoded_grid, dtype=np.uint8)
        decoded = Grid.decode(grid_array)
        grid_obj = decoded[0] if isinstance(decoded, tuple) else decoded
        if hasattr(base_env, "grid"):
            base_env.grid = grid_obj

    @staticmethod
    def _decode_carrying(carrying: dict[str, Any] | None):
        if carrying is None:
            return None

        encoded = carrying.get("encoded")
        if encoded is None:
            return None
        if isinstance(encoded, str):
            encoded = json.loads(encoded)
        encoded_tuple = tuple(int(v) for v in encoded)
        if len(encoded_tuple) != 3:
            return None

        obj = WorldObj.decode(*encoded_tuple)
        return obj
