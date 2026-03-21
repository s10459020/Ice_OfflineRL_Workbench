from typing import Any

import gymnasium as gym

from ice_offline.replay.state import State
from ice_offline.replay.state_io_wrapper import ensure_state_io


class StateRecordWrapper(gym.Wrapper):
    """Record current env state into info['state'] on reset/step."""

    def __init__(self, env: gym.Env):
        super().__init__(ensure_state_io(env))
        self._get_state = self.get_wrapper_attr("get_state")

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        state = self._get_state()
        info = dict(info)
        info["state"] = state.serialize_state()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = self._get_state()
        info = dict(info)
        info["state"] = state.serialize_state()
        return obs, reward, terminated, truncated, info


def ensure_state_record(env: gym.Env) -> gym.Env:
    current = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, StateRecordWrapper):
            return env
        current = current.env
    return StateRecordWrapper(env)
