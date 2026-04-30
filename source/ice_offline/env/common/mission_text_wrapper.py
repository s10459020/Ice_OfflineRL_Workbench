import gymnasium as gym
from gymnasium import spaces


DEFAULT_MISSION_CHARSET = " '(),0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz{}"
MISSION_MIN_LENGTH = 1
MISSION_MAX_LENGTH = 256


class MissionTextWrapper(gym.Wrapper):
    """Minigrid-only wrapper: set mission space to Text(1, 256, charset)."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        base = env.observation_space

        spaces_dict = dict(base.spaces)  # Minigrid dict space is assumed.
        spaces_dict["mission"] = spaces.Text(
            min_length=MISSION_MIN_LENGTH,
            max_length=MISSION_MAX_LENGTH,
            charset=DEFAULT_MISSION_CHARSET,
        )
        self.observation_space = spaces.Dict(spaces_dict)
