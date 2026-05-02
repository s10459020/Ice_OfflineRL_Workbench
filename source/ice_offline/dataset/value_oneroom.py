from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from ice_offline.dataset.value_collector import EvalFn, ValueCollector
from ice_offline.env.common import MissionTextWrapper, NoJpegImageWrapper


ENV_ID = "BabyAI-OneRoomS8-v0"
DIRECTIONS = (0, 1, 2, 3)
ACTIONS = (0, 1, 2, 3)


def make_value_env() -> gym.Env:
    env = FullyObsWrapper(gym.make(ENV_ID))
    env = MissionTextWrapper(env)
    env = NoJpegImageWrapper(env)
    return env


class ValueOneRoomCollector(ValueCollector):
    def __init__(
        self,
        env: gym.Env,
        agent: Any = None,
        *,
        eval_fn: EvalFn,
        output_file_name: str = "value_data.hdf5",
        output_dataset_key: str = "values",
    ) -> None:
        self._obs_samples: list[Any] | None = None
        super().__init__(
            env,
            agent=agent,
            eval_fn=eval_fn,
            output_file_name=output_file_name,
            output_dataset_key=output_dataset_key,
        )

    def _sample_obs(self, env: gym.Env) -> list[Any]:
        if self._obs_samples is None:
            width = env.width
            height = env.height
            samples: list[Any] = []
            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    for direction in DIRECTIONS:
                        old_pos = tuple(env.agent_pos)
                        old_dir = env.agent_dir
                        try:
                            env.agent_pos = (x, y)
                            env.agent_dir = direction
                            observation = env.gen_obs()
                        finally:
                            env.agent_pos = old_pos
                            env.agent_dir = old_dir
                        samples.append(observation)
            self._obs_samples = samples
        return self._obs_samples

    def _sample_act(self) -> list[int]:
        return list(ACTIONS)

    def _clear_cache(self) -> None:
        self._obs_samples = None
