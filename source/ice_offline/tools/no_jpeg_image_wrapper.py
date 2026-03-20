import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NoJpegImageWrapper(gym.ObservationWrapper):
    """Minigrid-only wrapper: force image dtype away from uint8."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        base = env.observation_space
        spaces_dict = dict(base.spaces)
        image_space = spaces_dict["image"]
        image_low = int(np.min(image_space.low))
        image_high = int(np.max(image_space.high))
        spaces_dict["image"] = spaces.Box(
            low=image_low,
            high=image_high,
            shape=image_space.shape,
            dtype=np.int16,
        )
        self.observation_space = spaces.Dict(spaces_dict)

    def observation(self, observation):
        obs = dict(observation)
        obs["image"] = np.asarray(obs["image"], dtype=np.int16)
        return obs
