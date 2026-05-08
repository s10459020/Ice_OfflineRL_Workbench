
import gymnasium as gym
import minari
import numpy as np

from ice_offline.data import State
from ice_offline.pipeline.state_loader import StateLoader
from ice_offline.env.common import StateIOWrapper


class StateInjectWrapper(gym.Wrapper):
    """Replay dataset transitions using recorded states/observations/actions."""

    # ====================
    # Init
    # ====================
    def __init__(
        self,
        env: gym.Env,
        dataset_id: str,
        *,
        random_episode: bool = False,
    ) -> None:
        self._state_io = StateIOWrapper(env)
        super().__init__(self._state_io)
        self.dataset = minari.load_dataset(dataset_id)
        self._state_loader = StateLoader(dataset_id)

        self.total_episodes = self.dataset.total_episodes
        self.random_episode = random_episode
        self._rng = np.random.default_rng()

        self._episode_index: int | None = None
        self._transition_index: int | None = None

        self._infos: list[dict] | None = None
        self._states: list[State] | None = None
        self._actions: list[int] | None = None
        self._rewards: list[float] | None = None
        self._observations: list | None = None
        self._transition_count: int | None = None

    # ====================
    # gym.Wrapper Overrides
    # ====================
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.env.reset(seed=seed, options=options)

        episode_index = self._pick_episode_index(options=options)
        self._load_episode_payload(episode_index)
        self._episode_index = episode_index

        state = self._states[0]
        self._state_io.set_state(state)
        self._transition_index = 0

        info = self._infos[0]
        observation = self._observations[0]
        return observation, info

    def step(self, action: int | None = None):
        curr_index = self._transition_index
        next_index = curr_index + 1
        if curr_index >= self._transition_count:
            return self._frozen_step()

        state = self._states[next_index]
        self._state_io.set_state(state)
        self._transition_index = next_index

        obs = self._observations[next_index]
        info = dict(self._infos[next_index])
        info["action"] = self._actions[curr_index]
        reward = self._rewards[curr_index]
        truncated = False
        terminated = next_index >= self._transition_count
        return obs, reward, terminated, truncated, info

    # ====================
    # Public API
    # ====================
    def close(self) -> None:
        self._state_loader.close()
        self.env.close()

    # ====================
    # Internal
    # ====================
    def _pick_episode_index(self, options: dict | None) -> int:
        if options and "episode_index" in options:
            episode_index = options["episode_index"]
            if episode_index < 0 or episode_index >= self.total_episodes:
                raise IndexError(f"episode_index out of range: {episode_index}")
            return episode_index

        if self.random_episode:
            return self._rng.integers(low=0, high=self.total_episodes)

        if self._episode_index is None:
            return 0
        return (self._episode_index + 1) % self.total_episodes

    def _frozen_step(self):
        transition_index = self._transition_index
        obs = self._observations[transition_index]
        info = self._infos[transition_index]
        return obs, 0.0, True, False, info

    def _load_episode_payload(self, episode_index: int) -> None:
        trajectory = self.dataset[episode_index]
        transition_count = len(trajectory.rewards)
        self._transition_count = transition_count
        self._actions = list(trajectory.actions)
        self._rewards = list(trajectory.rewards)
        self._observations = self._materialize_obs_seq(trajectory.observations, transition_count)
        self._infos = self._materialize_info_seq(trajectory.infos, transition_count)
        self._states = self._state_loader.load_episode(episode_index)

    def _materialize_obs_seq(self, observations, transition_count: int) -> list:
        return [{k: observations[k][i] for k in observations} for i in range(transition_count + 1)]

    def _materialize_info_seq(self, infos, transition_count: int) -> list[dict]:
        if infos is None:
            return [{} for _ in range(transition_count + 1)]
        return [self._index_payload(infos, i) for i in range(transition_count + 1)]

    def _index_payload(self, payload, index: int):
        if isinstance(payload, dict):
            return {k: self._index_payload(v, index) for k, v in payload.items()}
        return payload[index]
