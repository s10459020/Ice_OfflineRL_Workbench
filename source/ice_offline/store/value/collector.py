from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
from ice_offline.tools.paths import dataset_root


SampleBatch = list[Any]
EvalFn = Callable[[Any, Any, int], float]


class ValueCollector(gym.Wrapper):
    """Collector that executes injected sample/eval callbacks."""

    def __init__(
        self,
        env: gym.Env,
        agent: Any = None,
        *,
        eval_fn: EvalFn,
        output_file_name: str = "value_data.hdf5",
        output_dataset_key: str = "values",
    ) -> None:
        super().__init__(env)

        self._eval_fn = eval_fn
        self._agent = agent
        self._output_file_name = output_file_name
        self._output_dataset_key = output_dataset_key

        self._episodes: list[list[np.ndarray]] = []
        self._current: list[np.ndarray] | None = None
        self._last_values: np.ndarray | None = None
        self._observation_transforms: list[Callable[[Any], Any]] = []

    # ====================
    # override
    # ====================
    def reset(self, *args: Any, **kwargs: Any):
        observation, info = self.env.reset(*args, **kwargs)
        self._end_episode()
        self._current = []
        self._clear_cache()
        self._last_values = None
        return observation, info

    def step(self, action: int):
        return self.env.step(action)

    # ====================
    # public
    # ====================
    def eval(self) -> np.ndarray:
        samples = self._sample_obs(self.env.unwrapped)
        actions = self._sample_act()
        values = np.zeros((len(samples), len(actions)), dtype=np.float32)
        for sample_index, observation_raw in enumerate(samples):
            observation = self._transform_observation(observation_raw)
            for action_slot, action in enumerate(actions):
                values[sample_index, action_slot] = float(self._eval_fn(self._agent, observation, action))

        self._current.append(values)
        self._last_values = values
        return values

    def save(self, dataset_id: str) -> Path:
        self._end_episode()
        out_path = self._resolve_output_path(dataset_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(out_path, "w") as h5_file:
            for episode_index, sequence in enumerate(self._episodes):
                stacked = np.stack(sequence, axis=0)
                episode_group = h5_file.require_group(f"episode_{episode_index}")
                episode_group.create_dataset(self._output_dataset_key, data=stacked)
        return out_path

    def get_last(self) -> np.ndarray | None:
        return self._last_values

    # ====================
    # helping
    # ====================
    def _resolve_output_path(self, dataset_id: str) -> Path:
        return dataset_root() / dataset_id / "data" / self._output_file_name

    def _sample_obs(self, env: gym.Env) -> SampleBatch:
        raise NotImplementedError

    def _sample_act(self) -> list[int]:
        raise NotImplementedError

    def _clear_cache(self) -> None:
        raise NotImplementedError

    def _transform_observation(self, observation: Any) -> Any:
        if not self._observation_transforms:
            self._observation_transforms = self._build_observation_transforms()
        for transform in self._observation_transforms:
            observation = transform(observation)
        return observation

    def _build_observation_transforms(self) -> list[Callable[[Any], Any]]:
        wrappers: list[gym.Wrapper] = []
        current: gym.Env = self.env
        while isinstance(current, gym.Wrapper):
            wrappers.append(current)
            current = current.env
        transforms: list[Callable[[Any], Any]] = []
        for wrapper in reversed(wrappers):
            if isinstance(wrapper, gym.ObservationWrapper):
                transforms.append(wrapper.observation)
        return transforms

    def _end_episode(self) -> None:
        if self._current:
            self._episodes.append(self._current)
