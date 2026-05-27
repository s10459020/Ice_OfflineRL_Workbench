from typing import Type

import gymnasium as gym
import minari
import numpy as np

from ice_offline.pipeline.state._spec import StateIO
from ice_offline.pipeline.state.op_dataset import StateDataset


class StateInjectWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        dataset_id: str,
        state_cls: type,
        state_io_cls: Type[StateIO],
    ) -> None:
        super().__init__(env)
        self._state_io = state_io_cls(env)
        self._state_dataset = StateDataset.load_dataset(dataset_id=dataset_id, state_cls=state_cls)

        self.dataset = minari.load_dataset(dataset_id)
        self.total_episodes = self.dataset.total_episodes

        self._episode_index: int | None = None
        self._transition_index: int | None = None

        self._infos: list[dict] | None = None
        self._states: list | None = None
        self._actions: list[int] | None = None
        self._rewards: list[float] | None = None
        self._observations: list | None = None
        self._transition_count: int | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.env.reset(seed=seed, options=options)

        episode_index = self._select_episode_index(seed=seed, options=options)
        self._episode_index = episode_index
        self._load_episode_payload(episode_index)

        state = self._states[0]
        self._state_io.set_state(state)
        self._transition_index = 0
        return self._observations[0], self._infos[0]

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

    def close(self) -> None:
        self._state_dataset.close()
        self.env.close()

    def _select_episode_index(self, seed: int | None = None, options: dict | None = None) -> int:
        if options and "episode_index" in options:
            return options["episode_index"]
        if seed is not None:
            rng = np.random.default_rng(seed)
            return rng.integers(low=0, high=self.total_episodes)
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
        self._states = self._state_dataset.read_episode(episode_index)

    def _materialize_obs_seq(self, observations, transition_count: int) -> list:
        if isinstance(observations, dict):
            return [{k: observations[k][i] for k in observations} for i in range(transition_count + 1)]
        return [observations[i] for i in range(transition_count + 1)]

    def _materialize_info_seq(self, infos, transition_count: int) -> list[dict]:
        if infos is None:
            return [{} for _ in range(transition_count + 1)]
        return [self._index_payload(infos, i) for i in range(transition_count + 1)]

    def _index_payload(self, payload, index: int):
        if isinstance(payload, dict):
            return {k: self._index_payload(v, index) for k, v in payload.items()}
        return payload[index]


def make_replayer(
    dataset_id: str,
    state_cls: type,
    state_io_cls: Type[StateIO],
    eval_env: gym.Env | None = None,
):
    if not dataset_id.endswith("-v0"):
        raise ValueError(f"dataset_id must be full id with version suffix, got: {dataset_id}")
    if eval_env is None:
        dataset = minari.load_dataset(dataset_id)
        eval_env = gym.make(dataset.spec.env_spec.id, render_mode="human")
    return StateInjectWrapper(
        env=eval_env,
        dataset_id=dataset_id,
        state_cls=state_cls,
        state_io_cls=state_io_cls,
    )
