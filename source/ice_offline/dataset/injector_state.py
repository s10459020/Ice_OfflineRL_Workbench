
from typing import Any

import gymnasium as gym
import minari
import numpy as np

from ice_offline.data import State
from ice_offline.env.common import StateIOWrapper


class StateInjectWrapper(gym.Wrapper):
    """Replay dataset transitions using recorded states/observations/actions."""

    def __init__(
        self,
        env: gym.Env,
        dataset: str | Any,
        *,
        random_episode: bool = False,
    ) -> None:
        self._state_io = StateIOWrapper(env)
        super().__init__(self._state_io)
        self.dataset = minari.load_dataset(dataset) if isinstance(dataset, str) else dataset

        self.total_episodes = self.dataset.total_episodes
        self.random_episode = random_episode
        self._rng = np.random.default_rng()
        self._current_episode_pos: int | None = None

        self._state_index: int | None = None

        self._infos: list[dict[str, Any]] | None = None
        self._states: list[State] | None = None
        self._actions: list[int] | None = None
        self._rewards: list[float] | None = None
        self._transition_count: int | None = None
        self._observations: list[dict[str, Any]] | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.env.reset(seed=seed, options=options)

        episode_pos = self._pick_episode_pos(options=options)
        trajectory = self.dataset[episode_pos]
        self._current_episode_pos = episode_pos

        self._state_index = 0
        self._actions = list(trajectory.actions)
        self._rewards = list(trajectory.rewards)
        self._transition_count = len(self._rewards)
        self._observations = self._materialize_obs_seq(trajectory.observations, self._transition_count)
        self._infos = self._materialize_info_seq(trajectory.infos, self._transition_count)
        self._states = self._states_from_info(trajectory.infos, self._transition_count)

        state = self._states[0]
        self._state_io.set_state(state)

        observation = self._observations[0]
        info = self._infos[0]
        return observation, info

    def step(self, action: int | None = None):
        curr_index = self._state_index
        next_index = curr_index + 1
        if curr_index >= self._transition_count:
            return self._frozen_step()

        obs = self._observations[next_index]
        info = dict(self._infos[next_index])
        info["action"] = self._actions[curr_index]
        state = self._states[next_index]
        reward = self._rewards[curr_index]
        truncated = False
        terminated = next_index >= self._transition_count
        
        self._state_io.set_state(state)
        self._state_index = next_index
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    def _pick_episode_pos(self, options: dict[str, Any] | None) -> int:
        if options and "episode_index" in options:
            episode_pos = options["episode_index"]
            if episode_pos < 0 or episode_pos >= self.total_episodes:
                raise IndexError(f"episode_index out of range: {episode_pos}")
            return episode_pos

        if self.random_episode:
            return self._rng.integers(low=0, high=self.total_episodes)

        if self._current_episode_pos is None:
            return 0
        return (self._current_episode_pos + 1) % self.total_episodes

    def _frozen_step(self):
        state_index = self._state_index
        obs = self._observations[state_index]
        info = self._infos[state_index]
        return obs, 0.0, True, False, info

    def _states_from_info(self, infos: Any, transition_count: int) -> list[State]:
        state_payload = infos["state"]
        payload_seq = [{k: state_payload[k][i] for k in state_payload} for i in range(transition_count + 1)]
        return [State.from_serialized(payload) for payload in payload_seq]

    def _materialize_obs_seq(self, observations: Any, transition_count: int) -> list[dict[str, Any]]:
        return [{k: observations[k][i] for k in observations} for i in range(transition_count + 1)]

    def _materialize_info_seq(self, infos: Any, transition_count: int) -> list[dict[str, Any]]:
        return [self._index_payload(infos, i) for i in range(transition_count + 1)]

    def _index_payload(self, payload: Any, index: int) -> Any:
        if isinstance(payload, dict):
            return {k: self._index_payload(v, index) for k, v in payload.items()}
        return payload[index]
