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
    def __init__(self, path: Path | None, device: str = "cuda") -> None:
        self.path = Path(path) if path is not None else None
        self.device = device
        self.metadata_path = self.path.parent / "metadata.json" if self.path is not None else None
        metadata = self._read_metadata()
        self.dataset_id = metadata.get("dataset_id", "")

    # ====================
    # Loading
    # ====================
    def load_buffer(self) -> Buffer:
        if self._has_flat_buffer():
            return self._load_flat_buffer()
        return self.buffer_from_episodes(self.load_episodes(), device=self.device)

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
        env_id = self._read_env_id(metadata)
        if self._has_flat_buffer():
            return self.metadata_from_buffer(
                buffer=self.load_buffer(),
                env_id=env_id,
            )
        return self.metadata_from_episodes(
            episodes=self.load_episodes(),
            env_id=env_id,
        )

    # ====================
    # Episode helpers
    # ====================
    def buffer_from_episodes(self, episodes: list[Episode], device: str = "cuda") -> Buffer:
        obs_list: list[np.ndarray | dict[str, np.ndarray]] = []
        next_obs_list: list[np.ndarray | dict[str, np.ndarray]] = []
        act_list: list[np.ndarray] = []
        rew_list: list[np.ndarray] = []
        term_list: list[np.ndarray] = []
        trunc_list: list[np.ndarray] = []

        for episode in episodes:
            observations = episode.observations
            obs_list.append(self._slice_obs(observations, 0, -1))
            next_obs_list.append(self._slice_obs(observations, 1, None))
            act_list.append(np.asarray(episode.actions))
            rew_list.append(np.asarray(episode.rewards))
            term_list.append(np.asarray(episode.terminations))
            trunc_list.append(np.asarray(episode.truncations))

        term = np.concatenate(term_list, axis=0)
        trunc = np.concatenate(trunc_list, axis=0)
        return Buffer(
            observations=torch.as_tensor(self._concat_obs(obs_list), dtype=torch.float32, device=device),
            next_observations=torch.as_tensor(self._concat_obs(next_obs_list), dtype=torch.float32, device=device),
            actions=torch.as_tensor(np.concatenate(act_list, axis=0), dtype=torch.float32, device=device),
            rewards=torch.as_tensor(np.concatenate(rew_list, axis=0).reshape(-1, 1), dtype=torch.float32, device=device),
            dones=torch.as_tensor(np.logical_or(term, trunc).reshape(-1, 1), dtype=torch.float32, device=device),
        )

    def metadata_from_episodes(self, episodes: list[Episode], env_id: str) -> Metadata:
        first = episodes[0]
        observations = first.observations
        actions = np.asarray(first.actions)
        if isinstance(observations, dict):
            first_obs = next(iter(observations.values()))
            obs_shape = tuple(int(x) for x in first_obs.shape[1:])
        else:
            obs_shape = tuple(int(x) for x in observations.shape[1:])
        act_shape = tuple(int(x) for x in actions.shape[1:])
        count = int(sum(len(episode.rewards) for episode in episodes))
        return Metadata(
            env_id=env_id,
            obs_shape=obs_shape,
            act_shape=act_shape,
            obs_dim=int(np.prod(obs_shape)) if obs_shape else 1,
            act_dim=int(np.prod(act_shape)) if act_shape else 1,
            count=count,
        )

    def metadata_from_buffer(self, buffer: Buffer, env_id: str) -> Metadata:
        obs_shape = tuple(int(x) for x in buffer.observations.shape[1:])
        act_shape = tuple(int(x) for x in buffer.actions.shape[1:])
        return Metadata(
            env_id=env_id,
            obs_shape=obs_shape,
            act_shape=act_shape,
            obs_dim=int(np.prod(obs_shape)) if obs_shape else 1,
            act_dim=int(np.prod(act_shape)) if act_shape else 1,
            count=int(buffer.actions.shape[0]),
        )

    def write_episodes(self, path: Path, episodes: list[Episode]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h5_file:
            for index, episode in enumerate(episodes):
                episode_group = h5_file.create_group(f"episode_{index}")
                self.write_node(episode_group, "observations", episode.observations)
                self.write_node(episode_group, "actions", np.asarray(episode.actions))
                self.write_node(episode_group, "rewards", np.asarray(episode.rewards))
                self.write_node(episode_group, "terminations", np.asarray(episode.terminations, dtype=np.bool_))
                self.write_node(episode_group, "truncations", np.asarray(episode.truncations, dtype=np.bool_))
                self.write_node(episode_group, "infos", episode.infos or {})

    def write_buffer(self, path: Path, buffer: Buffer) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h5_file:
            self.write_node(h5_file, "observations", buffer.observations.detach().cpu().numpy())
            self.write_node(h5_file, "next_observations", buffer.next_observations.detach().cpu().numpy())
            self.write_node(h5_file, "actions", buffer.actions.detach().cpu().numpy())
            self.write_node(h5_file, "rewards", buffer.rewards.detach().cpu().numpy())
            self.write_node(h5_file, "dones", buffer.dones.detach().cpu().numpy())

    # ====================
    # HDF5 helpers
    # ====================
    def _read_metadata(self) -> dict:
        if self.metadata_path is None or not self.metadata_path.exists():
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

    def write_node(self, group: h5py.Group, name: str, value: np.ndarray | dict[str, object]) -> None:
        if isinstance(value, dict):
            child = group.create_group(name)
            for key, item in value.items():
                self.write_node(child, key, item)
            return
        group.create_dataset(name, data=np.asarray(value), compression="gzip")

    def _episode_keys_from_h5(self, h5_file: h5py.File) -> list[str]:
        return sorted(
            [key for key in h5_file.keys() if key.startswith("episode_")],
            key=lambda key: int(key.split("_")[1]),
        )

    def _has_flat_buffer(self) -> bool:
        if self.path is None or not self.path.exists():
            return False
        with h5py.File(self.path, "r") as h5_file:
            keys = set(h5_file.keys())
        return {
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "dones",
        }.issubset(keys)

    def _load_flat_buffer(self) -> Buffer:
        with h5py.File(self.path, "r") as h5_file:
            return Buffer(
                observations=torch.as_tensor(np.asarray(h5_file["observations"]), dtype=torch.float32, device=self.device),
                next_observations=torch.as_tensor(np.asarray(h5_file["next_observations"]), dtype=torch.float32, device=self.device),
                actions=torch.as_tensor(np.asarray(h5_file["actions"]), dtype=torch.float32, device=self.device),
                rewards=torch.as_tensor(np.asarray(h5_file["rewards"]), dtype=torch.float32, device=self.device),
                dones=torch.as_tensor(np.asarray(h5_file["dones"]), dtype=torch.float32, device=self.device),
            )
