from typing import Type

import gymnasium as gym
import numpy as np

from ice_offline.store.state._spec import StateIO
from ice_offline.store.state.op_dataset import StateDataset


class StateInjectWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        dataset,
        state_cls: type,
        state_io_cls: Type[StateIO],
    ) -> None:
        super().__init__(env)
        self.dataset = dataset
        self._state_io = state_io_cls(env)
        self._state_dataset = StateDataset.load_dataset(path=dataset.path.with_name("state_data.hdf5"), state_cls=state_cls)

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
        info = dict(self._infos[0])
        info["episode_index"] = episode_index
        return self._observations[0], info

    def step(self, action: int | None = None):
        curr_index = self._transition_index
        next_index = curr_index + 1
        if curr_index >= self._transition_count:
            return self._frozen_step()

        replay_action = self._actions[curr_index]
        self.env.step(replay_action)
        state = self._states[next_index]
        self._state_io.set_state(state)
        self._transition_index = next_index

        obs = self._observations[next_index]
        info = dict(self._infos[next_index])
        info["action"] = replay_action
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
            return rng.integers(low=0, high=self.dataset.episode_count)
        if self._episode_index is None:
            return 0
        return (self._episode_index + 1) % self.dataset.episode_count

    def _frozen_step(self):
        transition_index = self._transition_index
        obs = self._observations[transition_index]
        info = self._infos[transition_index]
        return obs, 0.0, True, False, info

    def _load_episode_payload(self, episode_index: int) -> None:
        trajectory = self.dataset.episodes[episode_index]
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
    dataset,
    state_cls: type,
    state_io_cls: Type[StateIO],
    eval_env: gym.Env | None = None,
    render_mode: str | None = "human",
):
    if eval_env is None:
        eval_env = gym.make(dataset.env_id, render_mode=render_mode)
    return StateInjectWrapper(
        env=eval_env,
        dataset=dataset,
        state_cls=state_cls,
        state_io_cls=state_io_cls,
    )


if __name__ == "__main__":
    from ice_offline.dataset._lookup import make_dataset
    from ice_offline.store.state._lookup import STATE_OPS
    from ice_offline.store.state.op_converter import StateConverter

    dataset = make_dataset("hopper_simple", device="cuda")
    state_cls, state_io_cls, converter_cls = STATE_OPS[dataset.env_id]
    StateConverter(dataset=dataset, converter_cls=converter_cls).convert()

    env = make_replayer(
        dataset=dataset,
        state_cls=state_cls,
        state_io_cls=state_io_cls,
        render_mode="human",
    )

    try:
        for episode in range(dataset.episode_count):
            env.reset(options={"episode_index": episode})
            env.render()
            while True:
                _, reward, terminated, truncated, info = env.step(None)
                env.render()
                if terminated or truncated:
                    break
    finally:
        env.close()
