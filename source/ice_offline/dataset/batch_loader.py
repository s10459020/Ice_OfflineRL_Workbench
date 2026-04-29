from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import minari
import numpy as np


ObsType = np.ndarray | dict[str, np.ndarray]
BatchType = dict[str, Any]
ObsTransform = Callable[[ObsType], np.ndarray]


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

    def sample_batch(self, batch_size: int, rng: np.random.Generator | None = None) -> BatchType:
        if rng is None:
            rng = np.random.default_rng(42)
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
    act_size: int
    buffer: TransitionBuffer
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    @property
    def num_transitions(self) -> int:
        return _obs_len(self.buffer.obs)

    @property
    def obs_size(self) -> int:
        return int(self.buffer.obs.shape[1])

    def sample_batch(self, batch_size: int, rng: np.random.Generator | None = None) -> BatchType:
        return self.buffer.sample_batch(batch_size=batch_size, rng=rng or self._rng)

    @staticmethod
    def _default_obs_transform(obs: ObsType) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32).reshape(_obs_len(obs), -1)

    @classmethod
    def _build_buffer_from_minari(
        cls,
        dataset_id: str,
        obs_transform: ObsTransform | None = None,
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

        obs_all = _concat_obs(obs_list)
        next_obs_all = _concat_obs(next_obs_list)
        obs_transform = obs_transform or cls._default_obs_transform
        obs_all = obs_transform(obs_all).astype(np.float32)
        next_obs_all = obs_transform(next_obs_all).astype(np.float32)
        act_all = np.concatenate(act_list, axis=0)
        rew_all = np.concatenate(rew_list, axis=0).astype(np.float32)
        done_all = np.concatenate(done_list, axis=0).astype(np.float32)

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
        obs_transform: ObsTransform | None = None,
        seed: int = 42,
    ) -> "BatchLoader":
        dataset = minari.load_dataset(dataset_id, download=True)
        action_space = dataset.spec.action_space
        if not hasattr(action_space, "n"):
            raise ValueError(f"Expected discrete action space, got: {type(action_space).__name__}")
        buffer = cls._build_buffer_from_minari(dataset_id, obs_transform=obs_transform)
        return cls(
            dataset_id=dataset_id,
            act_size=int(action_space.n),
            buffer=buffer,
            seed=seed,
        )


