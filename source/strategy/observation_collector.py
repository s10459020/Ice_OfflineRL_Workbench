from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces


def collect_observation_dataset(
    env: gym.Env,
    output_path: str | Path = "tmps/one_room_s8_data.hdf5",
    max_episodes: int = 3,
    max_episode_steps: int = 20,
    seed: int | None = 42,
    dataset_id: str = "one-room-s8-data-v0",
    record_infos: bool = True,
    overwrite_local_dataset: bool = True,
    normalize_mission_observation: bool = True,
) -> dict[str, Any]:
    """
    Collect Minari dataset with random policy and export main_data.hdf5 to output_path.
    """
    if max_episodes <= 0:
        return {"episodes": 0, "steps": 0, "path": str(Path(output_path))}
    if max_episode_steps <= 0:
        return {"episodes": 0, "steps": 0, "path": str(Path(output_path))}

    try:
        import minari
    except ImportError as exc:  # pragma: no cover
        raise ImportError("minari is required for observation collector.") from exc

    if overwrite_local_dataset:
        _try_delete_local_dataset(minari, dataset_id=dataset_id)

    if normalize_mission_observation:
        env = _normalize_mission_observation_if_needed(env)

    collector_env = _wrap_data_collector(minari, env=env, record_infos=record_infos)
    total_steps = 0
    episodes = 0

    try:
        for episode in range(1, max_episodes + 1):
            obs, info = collector_env.reset(seed=None if seed is None else seed + episode)
            _ = (obs, info)
            for _step in range(max_episode_steps):
                action = collector_env.action_space.sample()
                _, _, terminated, truncated, _ = collector_env.step(action)
                total_steps += 1
                if terminated or truncated:
                    break
            episodes += 1

        _create_dataset(minari, collector_env=collector_env, dataset_id=dataset_id)
    finally:
        collector_env.close()

    src_main_data = _resolve_main_data_path(dataset_id=dataset_id)
    dst_main_data = Path(output_path)
    dst_main_data.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_main_data, dst_main_data)

    return {
        "episodes": episodes,
        "steps": total_steps,
        "dataset_id": dataset_id,
        "minari_path": str(src_main_data),
        "path": str(dst_main_data),
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
    local_datasets = {}
    if hasattr(minari_module, "list_local_datasets"):
        local_datasets = minari_module.list_local_datasets()
    if dataset_id not in local_datasets:
        return
    if hasattr(minari_module, "delete_dataset"):
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
    mission_space = obs_space.spaces["mission"]
    if isinstance(mission_space, spaces.Text):
        return env
    return _NormalizeMissionObservationWrapper(env)


class _NormalizeMissionObservationWrapper(gym.ObservationWrapper):
    """Keep mission in observations but replace MissionSpace with a serializable Text space."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError("_NormalizeMissionObservationWrapper requires Dict observation space.")
        updated_spaces = dict(env.observation_space.spaces)
        updated_spaces["mission"] = spaces.Text(max_length=512)
        self.observation_space = spaces.Dict(updated_spaces)

    def observation(self, observation):
        if isinstance(observation, dict):
            normalized = dict(observation)
            if "mission" in normalized:
                normalized["mission"] = str(normalized["mission"])
            return normalized
        return observation
