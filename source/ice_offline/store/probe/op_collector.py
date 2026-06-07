from collections.abc import Callable
from pathlib import Path
from typing import Any, Type

import gymnasium as gym
import numpy as np

from ice_offline.store.probe.op_dataset import ProbeDataset


ProbeEvalFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


class ProbeInterface:
    state_io_cls: Type | None = None

    def set_env(self, env: gym.Env) -> None:
        pass

    def reset(self, state: Any) -> None:
        pass

    def step(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        pass

    def get_probes(self, observation: Any) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class ProbeCollectWrapper(gym.Wrapper):
    # ====================
    # Init
    # ====================
    def __init__(
        self,
        env: gym.Env,
        probe: ProbeInterface,
        eval_fn: ProbeEvalFn,
    ) -> None:
        super().__init__(env)
        self._probe = probe
        self._probe.set_env(env)
        self._eval_fn = eval_fn
        state_io_cls = self._probe.state_io_cls
        self._state_io = state_io_cls(env) if state_io_cls is not None else None
        self._episodes: list[list[dict[str, np.ndarray]]] = []
        self._episode: list[dict[str, np.ndarray]] | None = None
        self._last_state: Any = None

    # ====================
    # Public API
    # ====================
    def save(self, path: Path) -> ProbeDataset:
        self._end_episode()
        return ProbeDataset.write(path=path.with_name("probe_data.hdf5"), episodes=self._episodes)

    # ====================
    # Overwrite
    # ====================
    def reset(self, *args: Any, **kwargs: Any):
        observation, info = self.env.reset(*args, **kwargs)
        self._end_episode()
        state = self._state(info, observation)
        self._probe.reset(state)
        self._last_state = state
        self._episode = []
        self._record(observation)
        return observation, info

    def step(self, *args: Any, **kwargs: Any):
        action = args[0] if args else kwargs.get("action")
        observation, reward, terminated, truncated, info = self.env.step(*args, **kwargs)
        state = self._state(info, observation)
        done = bool(terminated or truncated)
        self._probe.step(self._last_state, action, float(reward), state, done)
        self._last_state = state
        self._record(observation)
        return observation, reward, terminated, truncated, info

    # ====================
    # Private
    # ====================
    def _record(self, observation: Any) -> dict[str, np.ndarray]:
        probe_observations, probe_actions = self._probe.get_probes(observation)
        values = self._eval_fn(probe_observations, probe_actions)
        payload = {
            "observations": np.asarray(probe_observations),
            "actions": np.asarray(probe_actions),
            "values": np.asarray(values),
        }
        self._episode.append(payload)
        return payload

    def _state(self, info: dict[str, Any], observation: Any) -> Any:
        if self._state_io is not None:
            return self._state_io.get_state()
        return info.get("state", observation)

    def _end_episode(self) -> None:
        if self._episode:
            self._episodes.append(self._episode)
