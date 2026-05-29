import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


class MinariEpisode:
    def __init__(
        self,
        observations: Any,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
        infos: Any,
    ) -> None:
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.terminations = terminations
        self.truncations = truncations
        self.infos = infos


class MinariLoader:
    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)
        self.metadata_path = self.dataset_path.parent / "metadata.json"
        self.metadata = self._read_metadata(self.metadata_path)
        self.dataset_id = self.metadata.get("dataset_id")
        self.env_id = self._read_env_id(self.metadata)

        self.episodes, self.buffer = self._load_data(self.dataset_path)
        self._episode_keys = sorted(self.episodes.keys(), key=lambda key: int(key.split("_")[1]))
        self.total_episodes = len(self._episode_keys)
        self.episode_steps = [int(self.episodes[key]["rewards"].shape[0]) for key in self._episode_keys]
        self.total_steps = int(sum(self.episode_steps))

    def __getitem__(self, episode_index: int) -> MinariEpisode:
        episode = self.episodes[self._episode_keys[episode_index]]
        return MinariEpisode(
            observations=episode["observations"],
            actions=episode["actions"],
            rewards=episode["rewards"],
            terminations=episode["terminations"],
            truncations=episode["truncations"],
            infos=episode["infos"],
        )

    def iterate_episodes(self):
        for episode_key in self._episode_keys:
            episode = self.episodes[episode_key]
            yield MinariEpisode(
                observations=episode["observations"],
                actions=episode["actions"],
                rewards=episode["rewards"],
                terminations=episode["terminations"],
                truncations=episode["truncations"],
                infos=episode["infos"],
            )

    def _load_data(self, dataset_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray | dict[str, np.ndarray]]]:
        print(f"[data] loading: {dataset_path}")
        episodes: dict[str, dict[str, Any]] = {}
        obs_list: list[np.ndarray | dict[str, np.ndarray]] = []
        next_obs_list: list[np.ndarray | dict[str, np.ndarray]] = []
        act_list: list[np.ndarray] = []
        rew_list: list[np.ndarray] = []
        term_list: list[np.ndarray] = []
        trunc_list: list[np.ndarray] = []

        with h5py.File(dataset_path, "r") as h5_file:
            episode_keys = self._episode_keys_from_h5(h5_file)
            total = len(episode_keys)
            print(f"[data] episodes={total}")
            for i, episode_key in enumerate(episode_keys):
                episode = h5_file[episode_key]

                observations = self._read_node(episode["observations"])
                actions = np.asarray(episode["actions"])
                rewards = np.asarray(episode["rewards"])
                terminations = np.asarray(episode["terminations"])
                truncations = np.asarray(episode["truncations"])
                infos = self._read_node(episode["infos"])

                episodes[episode_key] = {
                    "observations": observations,
                    "actions": actions,
                    "rewards": rewards,
                    "terminations": terminations,
                    "truncations": truncations,
                    "infos": infos,
                }

                obs_list.append(self._slice_obs(observations, 0, -1))
                next_obs_list.append(self._slice_obs(observations, 1, None))
                act_list.append(actions)
                rew_list.append(rewards)
                term_list.append(terminations)
                trunc_list.append(truncations)
                if (i + 1) % 100 == 0 or (i + 1) == total:
                    print(f"[data] loaded {i + 1}/{total} episodes")

        print("[data] concatenating buffer arrays...")
        buffer = {
            "observations": self._concat_obs(obs_list),
            "next_observations": self._concat_obs(next_obs_list),
            "actions": np.concatenate(act_list, axis=0),
            "rewards": np.concatenate(rew_list, axis=0),
            "terminations": np.concatenate(term_list, axis=0),
            "truncations": np.concatenate(trunc_list, axis=0),
        }
        print("[data] buffer ready")
        return episodes, buffer

    def _read_metadata(self, metadata_path: Path) -> dict:
        if not metadata_path.exists():
            print(f"[metadata] missing: {metadata_path}")
            return {}
        print(f"[metadata] loading: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        decoded = self._decode_metadata(metadata, path="$", depth=0)
        print("[metadata] decode done")
        return decoded

    def _decode_metadata(self, value, path: str, depth: int):
        print(f"[metadata] depth={depth} path={path} type={type(value).__name__}")
        if isinstance(value, dict):
            return {
                key: self._decode_metadata(item, path=f"{path}.{key}", depth=depth + 1)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self._decode_metadata(item, path=f"{path}[{i}]", depth=depth + 1)
                for i, item in enumerate(value)
            ]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return value
            print(f"[metadata] parsed-json path={path}")
            return self._decode_metadata(parsed, path=path, depth=depth + 1)
        return value

    def _read_env_id(self, metadata: dict) -> str | None:
        env_spec = metadata.get("env_spec")
        if isinstance(env_spec, dict):
            return env_spec.get("id")
        if isinstance(env_spec, str):
            parsed = json.loads(env_spec)
            return parsed.get("id")
        return None

    def _read_node(self, node):
        if isinstance(node, h5py.Dataset):
            return np.asarray(node)
        return {key: self._read_node(node[key]) for key in node.keys()}

    def _slice_obs(self, obs_all, start: int, stop: int | None):
        if isinstance(obs_all, dict):
            return {key: value[start:stop] for key, value in obs_all.items()}
        return obs_all[start:stop]

    def _concat_obs(self, obs_list: list[np.ndarray | dict[str, np.ndarray]]):
        first = obs_list[0]
        if isinstance(first, dict):
            return {
                key: np.concatenate([obs[key] for obs in obs_list], axis=0)
                for key in first.keys()
            }
        return np.concatenate(obs_list, axis=0)

    def _episode_keys_from_h5(self, h5_file: h5py.File) -> list[str]:
        return sorted(
            [key for key in h5_file.keys() if key.startswith("episode_")],
            key=lambda key: int(key.split("_")[1]),
        )
