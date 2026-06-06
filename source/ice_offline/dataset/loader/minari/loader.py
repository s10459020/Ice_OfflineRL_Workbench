import json
from pathlib import Path

import h5py
import numpy as np
import torch

from ice_offline.dataset._types import Buffer, Episode, Metadata


class MinariLoader:
    # ====================
    # Init
    # ====================
    def __init__(self, path: Path, device: str = "cpu") -> None:
        self.path = Path(path)
        self.device = device
        self.metadata_path = self.path.parent / "metadata.json"
        metadata = self._read_metadata()
        self.dataset_id = metadata.get("dataset_id", "")

    # ====================
    # Loading
    # ====================
    def load_buffer(self) -> Buffer:
        obs_list: list[np.ndarray | dict[str, np.ndarray]] = []
        next_obs_list: list[np.ndarray | dict[str, np.ndarray]] = []
        act_list: list[np.ndarray] = []
        rew_list: list[np.ndarray] = []
        term_list: list[np.ndarray] = []
        trunc_list: list[np.ndarray] = []

        with h5py.File(self.path, "r") as h5_file:
            episode_keys = self._episode_keys_from_h5(h5_file)
            for episode_key in episode_keys:
                episode = h5_file[episode_key]
                observations = self._read_node(episode["observations"])
                actions = np.asarray(episode["actions"])
                rewards = np.asarray(episode["rewards"])
                terminations = np.asarray(episode["terminations"])
                truncations = np.asarray(episode["truncations"])

                obs_list.append(self._slice_obs(observations, 0, -1))
                next_obs_list.append(self._slice_obs(observations, 1, None))
                act_list.append(actions)
                rew_list.append(rewards)
                term_list.append(terminations)
                trunc_list.append(truncations)

        term = np.concatenate(term_list, axis=0)
        trunc = np.concatenate(trunc_list, axis=0)
        buffer = Buffer(
            observations=torch.as_tensor(self._concat_obs(obs_list), dtype=torch.float32, device=self.device),
            next_observations=torch.as_tensor(self._concat_obs(next_obs_list), dtype=torch.float32, device=self.device),
            actions=torch.as_tensor(np.concatenate(act_list, axis=0), dtype=torch.float32, device=self.device),
            rewards=torch.as_tensor(np.concatenate(rew_list, axis=0).reshape(-1, 1), dtype=torch.float32, device=self.device),
            dones=torch.as_tensor(np.logical_or(term, trunc).reshape(-1, 1), dtype=torch.float32, device=self.device),
        )
        return buffer

    def load_episodes(self) -> list[Episode]:
        episodes: list[Episode] = []
        with h5py.File(self.path, "r") as h5_file:
            episode_keys = self._episode_keys_from_h5(h5_file)
            for episode_key in episode_keys:
                episode = h5_file[episode_key]
                episodes.append(
                    Episode(
                        observations=self._read_node(episode["observations"]),
                        actions=np.asarray(episode["actions"]),
                        rewards=np.asarray(episode["rewards"]),
                        terminations=np.asarray(episode["terminations"]),
                        truncations=np.asarray(episode["truncations"]),
                        infos=self._read_node(episode["infos"]),
                    )
                )

        return episodes

    # ====================
    # Metadata
    # ====================
    def load_metadata(self) -> Metadata:
        metadata = self._read_metadata()
        with h5py.File(self.path, "r") as h5_file:
            episode_keys = self._episode_keys_from_h5(h5_file)
            episode_steps = [int(h5_file[key]["rewards"].shape[0]) for key in episode_keys]
            count = int(sum(episode_steps))

            if episode_keys:
                first = h5_file[episode_keys[0]]
                observations = self._read_node(first["observations"])
                actions = np.asarray(first["actions"])
                if isinstance(observations, dict):
                    first_obs = next(iter(observations.values()))
                    obs_shape = tuple(int(x) for x in first_obs.shape[1:])
                else:
                    obs_shape = tuple(int(x) for x in observations.shape[1:])
                act_shape = tuple(int(x) for x in actions.shape[1:])
            else:
                obs_shape = ()
                act_shape = ()

        return Metadata(
            env_id=self._read_env_id(metadata),
            obs_shape=obs_shape,
            act_shape=act_shape,
            obs_dim=int(np.prod(obs_shape)) if obs_shape else 1,
            act_dim=int(np.prod(act_shape)) if act_shape else 1,
            count=count,
        )

    # ====================
    # HDF5 helpers
    # ====================
    def _read_metadata(self) -> dict:
        if not self.metadata_path.exists():
            return {}
        with self.metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _read_env_id(self, metadata: dict) -> str:
        if "env_id" in metadata:
            return metadata["env_id"]
        env_spec = metadata.get("env_spec", {})
        if isinstance(env_spec, str):
            env_spec = json.loads(env_spec)
        return env_spec.get("id", "")

    def _read_node(self, node):
        if isinstance(node, h5py.Dataset):
            return np.asarray(node)
        return {key: self._read_node(node[key]) for key in node.keys()}

    def _slice_obs(self, observations, start: int, stop: int | None):
        if isinstance(observations, dict):
            return {key: value[start:stop] for key, value in observations.items()}
        return observations[start:stop]

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
