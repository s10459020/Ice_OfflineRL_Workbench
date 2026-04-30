
from dataclasses import dataclass
from typing import Any, Callable

import minari
import numpy as np


ObsType = np.ndarray | dict[str, np.ndarray]
BatchType = dict[str, Any]
EncodeFn = Callable[[ObsType], np.ndarray]


def _slice_obs(obs: ObsType, idx: np.ndarray) -> ObsType:
    if isinstance(obs, dict):
        return {k: v[idx] for k, v in obs.items()}
    return obs[idx]


def _slice_obs_range(obs: ObsType, start: int | None, end: int | None) -> ObsType:
    if isinstance(obs, dict):
        return {k: v[start:end] for k, v in obs.items()}
    return obs[start:end]


def _concat_obs(obs_list: list[ObsType]) -> ObsType:
    first = obs_list[0]
    if isinstance(first, dict):
        return {k: np.concatenate([obs[k] for obs in obs_list], axis=0) for k in first}
    return np.concatenate(obs_list, axis=0)


def _obs_len(obs: ObsType) -> int:
    if isinstance(obs, dict):
        first_key = next(iter(obs))
        return int(obs[first_key].shape[0])
    return int(obs.shape[0])


@dataclass
class TransitionBuffer:
    obs: np.ndarray
    act: np.ndarray
    rew: np.ndarray
    next_obs: np.ndarray
    done: np.ndarray

    def sample_batch(self, batch_size: int, seed: int | None = None) -> BatchType:
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng(42)
        idx = rng.integers(0, _obs_len(self.obs), size=(batch_size,))
        return {
            "obs": _slice_obs(self.obs, idx),
            "act": self.act[idx],
            "rew": self.rew[idx],
            "next_obs": _slice_obs(self.next_obs, idx),
            "done": self.done[idx],
        }


@dataclass
class BatchLoader:
    dataset_id: str
    obs_shape: tuple[int, ...]
    obs_size: int
    act_shape: tuple[int, ...]
    act_size: int
    buffer: TransitionBuffer

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(42)

    @property
    def num_transitions(self) -> int:
        return _obs_len(self.buffer.obs)

    def sample_batch(self, batch_size: int, seed: int | None = None) -> BatchType:
        if seed is not None:
            return self.buffer.sample_batch(batch_size=batch_size, seed=seed)
        idx = self._rng.integers(0, _obs_len(self.buffer.obs), size=(batch_size,))
        return {
            "obs": _slice_obs(self.buffer.obs, idx),
            "act": self.buffer.act[idx],
            "rew": self.buffer.rew[idx],
            "next_obs": _slice_obs(self.buffer.next_obs, idx),
            "done": self.buffer.done[idx],
        }

    @classmethod
    def _build_buffer_from_minari(
        cls,
        dataset_id: str,
        obs_encode: EncodeFn | None = None,
        act_encode: EncodeFn | None = None,
    ) -> TransitionBuffer:
        dataset = minari.load_dataset(dataset_id, download=True)

        obs_list: list[ObsType] = []
        act_list: list[np.ndarray] = []
        rew_list: list[np.ndarray] = []
        next_obs_list: list[ObsType] = []
        done_list: list[np.ndarray] = []

        for episode in dataset.iterate_episodes():
            obs_raw = episode.observations
            if isinstance(obs_raw, dict):
                obs: ObsType = {k: np.asarray(v) for k, v in obs_raw.items()}
            else:
                obs = np.asarray(obs_raw)
            act = np.asarray(episode.actions)
            rew = np.asarray(episode.rewards, dtype=np.float32)
            term = np.asarray(episode.terminations, dtype=np.float32)
            trunc = np.asarray(episode.truncations, dtype=np.float32)

            done = np.clip(term + trunc, 0.0, 1.0)
            obs_list.append(_slice_obs_range(obs, 0, -1))
            next_obs_list.append(_slice_obs_range(obs, 1, None))
            act_list.append(act)
            rew_list.append(rew)
            done_list.append(done)

        obs_encode = obs_encode or (lambda obs: np.asarray(obs, dtype=np.float32))
        act_encode = act_encode or (lambda act: np.asarray(act))

        obs_all = _concat_obs(obs_list)
        next_obs_all = _concat_obs(next_obs_list)
        act_all = np.concatenate(act_list, axis=0)
        rew_all = np.concatenate(rew_list, axis=0)
        done_all = np.concatenate(done_list, axis=0)

        obs_all = obs_encode(obs_all).astype(np.float32)
        next_obs_all = obs_encode(next_obs_all).astype(np.float32)
        act_all = act_encode(act_all)
        rew_all = rew_all.astype(np.float32)
        done_all = done_all.astype(np.float32)

        return TransitionBuffer(
            obs=obs_all,
            act=act_all,
            rew=rew_all,
            next_obs=next_obs_all,
            done=done_all,
        )

    @classmethod
    def from_minari(
        cls,
        dataset_id: str,
        obs_encode: EncodeFn | None = None,
        act_encode: EncodeFn | None = None,
    ) -> "BatchLoader":
        dataset = minari.load_dataset(dataset_id, download=True)
        action_space = dataset.spec.action_space
        observation_space = dataset.spec.observation_space
        buffer = cls._build_buffer_from_minari(
            dataset_id,
            obs_encode=obs_encode,
            act_encode=act_encode,
        )
        obs_arr = np.asarray(buffer.obs)
        act_arr = np.asarray(buffer.act)
        obs_shape = tuple(int(x) for x in obs_arr.shape[1:]) if obs_arr.ndim > 1 else ()
        act_shape = tuple(int(x) for x in act_arr.shape[1:]) if act_arr.ndim > 1 else ()
        obs_size = int(getattr(observation_space, "n", 0))
        act_size = int(getattr(action_space, "n", 0))
        return cls(
            dataset_id=dataset_id,
            obs_shape=obs_shape,
            obs_size=obs_size,
            act_shape=act_shape,
            act_size=act_size,
            buffer=buffer,
        )


