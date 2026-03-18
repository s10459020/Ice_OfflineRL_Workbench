
import os
from pathlib import Path
import shutil
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ObservationCollector:
    """Collect Minari observation dataset and copy main_data.hdf5 to target path."""

    def __init__(
        self,
        output_path: str | Path,
        dataset_id: str,
        *,
        record_infos: bool = True,
        overwrite_local_dataset: bool = True,
        normalize_mission_observation: bool = True,
        preserve_image_values: bool = True,
    ) -> None:
        try:
            import minari
        except ImportError as exc:  # pragma: no cover
            raise ImportError("minari is required for observation collector.") from exc

        self._minari = minari
        self.output_path = Path(output_path)
        self.dataset_id = str(dataset_id)
        self.record_infos = bool(record_infos)
        self.overwrite_local_dataset = bool(overwrite_local_dataset)
        self.normalize_mission_observation = bool(normalize_mission_observation)
        self.preserve_image_values = bool(preserve_image_values)
        self._collector_env: gym.Env | None = None

    def prepare_env(self, env: gym.Env) -> gym.Env:
        if self.overwrite_local_dataset:
            _try_delete_local_dataset(self._minari, dataset_id=self.dataset_id)
        wrapped = _normalize_mission_observation_if_needed(env) if self.normalize_mission_observation else env
        if self.preserve_image_values:
            wrapped = _NoJpegImageObservationWrapper(wrapped)
        wrapped = _DropStateInfoWrapper(wrapped)
        self._collector_env = _wrap_data_collector(self._minari, env=wrapped, record_infos=self.record_infos)
        return self._collector_env

    def on_reset(self, info: dict[str, Any]) -> None:
        _ = info

    def on_step(
        self,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        _ = (action, reward, terminated, truncated, info)

    def on_episode_end(self, *, forced_cutoff: bool) -> None:
        _ = forced_cutoff

    def close(self) -> None:
        if self._collector_env is None:
            return
        try:
            _create_dataset(self._minari, collector_env=self._collector_env, dataset_id=self.dataset_id)
        finally:
            self._collector_env.close()
        src_main_data = _resolve_main_data_path(dataset_id=self.dataset_id)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_main_data, self.output_path)

    def result(self) -> dict[str, Any]:
        src_main_data = _resolve_main_data_path(dataset_id=self.dataset_id)
        return {
            "dataset_id": self.dataset_id,
            "minari_path": str(src_main_data),
            "path": str(self.output_path),
        }


def _wrap_data_collector(minari_module: Any, env: gym.Env, record_infos: bool):
    collector_cls = getattr(minari_module, "DataCollector", None)
    if collector_cls is not None:
        return collector_cls(env, record_infos=record_infos)
    collector_cls = getattr(minari_module, "DataCollectorV0", None)
    if collector_cls is not None:
        return collector_cls(env, record_infos=record_infos)
    raise RuntimeError("Unsupported minari version: missing DataCollector/DataCollectorV0.")


def _create_dataset(minari_module: Any, collector_env: gym.Env, dataset_id: str) -> None:
    if hasattr(collector_env, "create_dataset"):
        collector_env.create_dataset(dataset_id=dataset_id)
        return
    if hasattr(minari_module, "create_dataset_from_collector_env"):
        minari_module.create_dataset_from_collector_env(dataset_id=dataset_id, collector_env=collector_env)
        return
    raise RuntimeError("Unsupported minari version: no dataset creation API found.")


def _try_delete_local_dataset(minari_module: Any, dataset_id: str) -> None:
    local_datasets = minari_module.list_local_datasets() if hasattr(minari_module, "list_local_datasets") else {}
    if dataset_id in local_datasets and hasattr(minari_module, "delete_dataset"):
        minari_module.delete_dataset(dataset_id)


def _resolve_main_data_path(dataset_id: str) -> Path:
    root = Path(os.environ.get("MINARI_DATASETS_PATH", Path.home() / ".minari" / "datasets"))
    return root / dataset_id / "data" / "main_data.hdf5"


def _normalize_mission_observation_if_needed(env: gym.Env) -> gym.Env:
    obs_space = env.observation_space
    if not isinstance(obs_space, spaces.Dict):
        return env
    if "mission" not in obs_space.spaces:
        return env
    if isinstance(obs_space.spaces["mission"], spaces.Text):
        return env
    return _NormalizeMissionObservationWrapper(env)


class _NormalizeMissionObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError("_NormalizeMissionObservationWrapper requires Dict observation space.")
        updated_spaces = dict(env.observation_space.spaces)
        updated_spaces["mission"] = spaces.Text(max_length=512)
        self.observation_space = spaces.Dict(updated_spaces)

    def observation(self, observation):
        if isinstance(observation, dict) and "mission" in observation:
            normalized = dict(observation)
            normalized["mission"] = str(normalized["mission"])
            return normalized
        return observation


class _DropStateInfoWrapper(gym.Wrapper):
    """Drop replay-only info['state'] so Minari info serialization stays HDF5-safe."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(info, dict) and "state" in info:
            info = dict(info)
            info.pop("state", None)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(info, dict) and "state" in info:
            info = dict(info)
            info.pop("state", None)
        return obs, reward, terminated, truncated, info


class _NoJpegImageObservationWrapper(gym.ObservationWrapper):
    """
    Minari HDF5 path JPEG-encodes uint8 images.
    Cast image to int16 so encoded grids are stored losslessly.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        if not isinstance(obs_space, spaces.Dict) or "image" not in obs_space.spaces:
            self._enabled = False
            self.observation_space = obs_space
            return
        image_space = obs_space.spaces["image"]
        if not isinstance(image_space, spaces.Box):
            self._enabled = False
            self.observation_space = obs_space
            return
        if image_space.dtype != np.uint8:
            self._enabled = False
            self.observation_space = obs_space
            return
        updated_spaces = dict(obs_space.spaces)
        updated_spaces["image"] = spaces.Box(low=0, high=255, shape=image_space.shape, dtype=np.int16)
        self.observation_space = spaces.Dict(updated_spaces)
        self._enabled = True

    def observation(self, observation):
        if not self._enabled or not isinstance(observation, dict) or "image" not in observation:
            return observation
        normalized = dict(observation)
        normalized["image"] = np.asarray(normalized["image"], dtype=np.int16)
        return normalized
