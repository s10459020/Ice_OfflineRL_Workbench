from __future__ import annotations

import json
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .read_state_dataset import StateDatasetReader
from .state_types import AgentState


class StateReplayWrapper(gym.Wrapper):
    """
    Replay recorded trajectory states as a deterministic fake env rollout.

    reset():
      - selects one episode from the dataset
      - returns the first recorded state as observation

    step(action):
      - action is ignored by default
      - advances to the next recorded state
      - episode terminates at last recorded state
    """

    def __init__(
        self,
        env: gym.Env,
        reader: StateDatasetReader,
        *,
        random_episode: bool = False,
        strict_action_check: bool = False,
    ) -> None:
        super().__init__(env)
        self.reader = reader
        self.random_episode = bool(random_episode)
        self.strict_action_check = bool(strict_action_check)

        first_state = self.reader.get_state(episode_index=0, state_index=0)
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
        self._episode_length = self.reader.episode_length(episode_index)

        state = self.reader.get_state(episode_index=episode_index, state_index=0)
        self._apply_state_to_env(state)
        obs = self._state_to_obs(state)

        info = dict(base_info)
        info["state"] = state
        info["episode_index"] = episode_index
        info["state_index"] = 0
        info["trajectory_length"] = self._episode_length
        return obs, info

    def step(self, action):
        self._ensure_active_episode()
        if self.strict_action_check and action is not None:
            raise RuntimeError("strict_action_check=True but action labels are not stored in dataset.")

        next_state_index = int(self._state_index) + 1
        episode_index = int(self._current_episode_index)
        if next_state_index >= int(self._episode_length):
            # No more recorded transitions: keep the current state unchanged
            # until caller triggers the next reset().
            state = self.reader.get_state(episode_index=episode_index, state_index=int(self._state_index))
            obs = self._state_to_obs(state)
            reward = 0.0
            terminated = True
            truncated = False
            info = {
                "state": state,
                "episode_index": episode_index,
                "state_index": int(self._state_index),
                "trajectory_length": int(self._episode_length),
                "action_ignored": True,
                "replay_frozen": True,
            }
            return obs, reward, terminated, truncated, info

        state = self.reader.get_state(episode_index=episode_index, state_index=next_state_index)
        self._state_index = next_state_index
        self._apply_state_to_env(state)

        obs = self._state_to_obs(state)
        reward = 0.0
        terminated = next_state_index == int(self._episode_length) - 1
        truncated = False
        info = {
            "state": state,
            "episode_index": episode_index,
            "state_index": next_state_index,
            "trajectory_length": int(self._episode_length),
            "action_ignored": True,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    def _pick_episode_index(self, options: dict[str, Any] | None) -> int:
        if options and "episode_index" in options:
            episode_index = int(options["episode_index"])
            if episode_index < 0 or episode_index >= self.reader.num_episodes:
                raise IndexError(f"episode_index out of range: {episode_index}")
            return episode_index

        if self.random_episode:
            return int(self._rng.integers(low=0, high=self.reader.num_episodes))

        if self._current_episode_index is None:
            return 0
        return (int(self._current_episode_index) + 1) % self.reader.num_episodes

    def _ensure_active_episode(self) -> None:
        if self._current_episode_index is None or self._state_index is None or self._episode_length is None:
            raise RuntimeError("Call reset() before step().")

    @staticmethod
    def _state_to_obs(state: AgentState) -> dict[str, Any]:
        return {
            "image": np.asarray(state.grid, dtype=np.uint8).copy(),
            "direction": int(state.agent_dir),
            "mission": str(state.mission),
        }

    def _apply_state_to_env(self, state: AgentState) -> None:
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
        try:
            from minigrid.core.grid import Grid
        except ImportError as exc:  # pragma: no cover
            raise ImportError("minigrid is required to restore grid state during replay.") from exc

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

        try:
            from minigrid.core.world_object import WorldObj
        except ImportError:
            return None

        obj = WorldObj.decode(*encoded_tuple)
        return obj
