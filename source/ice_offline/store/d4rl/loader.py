from pathlib import Path

import h5py
import numpy as np
import torch
import json

from ice_offline.dataset._types import Buffer, Episode, Metadata
from ice_offline.store.d4rl._lookup import D4RL_ENV_IDS


class D4rlLoader:
    def __init__(self, path: Path, device: str = "cuda") -> None:
        self.path = Path(path)
        self.device = device

    def load_buffer(self) -> Buffer:
        with h5py.File(self.path, "r") as h5_file:
            terminations = np.asarray(h5_file["terminals"], dtype=np.bool_)
            truncations = np.asarray(h5_file["timeouts"], dtype=np.bool_)
            return Buffer(
                observations=torch.as_tensor(np.asarray(h5_file["observations"]), dtype=torch.float32, device=self.device),
                next_observations=torch.as_tensor(np.asarray(h5_file["next_observations"]), dtype=torch.float32, device=self.device),
                actions=torch.as_tensor(np.asarray(h5_file["actions"]), dtype=torch.float32, device=self.device),
                rewards=torch.as_tensor(np.asarray(h5_file["rewards"]).reshape(-1, 1), dtype=torch.float32, device=self.device),
                dones=torch.as_tensor(np.logical_or(terminations, truncations).reshape(-1, 1), dtype=torch.float32, device=self.device),
            )

    def load_episodes(self) -> list[Episode]:
        episodes: list[Episode] = []
        with h5py.File(self.path, "r") as h5_file:
            observations = np.asarray(h5_file["observations"])
            next_observations = np.asarray(h5_file["next_observations"])
            actions = np.asarray(h5_file["actions"])
            rewards = np.asarray(h5_file["rewards"])
            terminations = np.asarray(h5_file["terminals"])
            truncations = np.asarray(h5_file["timeouts"])
            dones = np.logical_or(terminations, truncations)

            start = 0
            for index, done in enumerate(dones):
                if done:
                    stop = index + 1
                    episodes.append(
                        Episode(
                            observations=np.concatenate(
                                [observations[start:stop], next_observations[index:index + 1]],
                                axis=0,
                            ),
                            actions=actions[start:stop],
                            rewards=rewards[start:stop],
                            terminations=terminations[start:stop],
                            truncations=truncations[start:stop],
                            infos=None,
                        )
                    )
                    start = stop

            if start < len(dones):
                episodes.append(
                    Episode(
                        observations=np.concatenate(
                            [observations[start:], next_observations[-1:]],
                            axis=0,
                        ),
                        actions=actions[start:],
                        rewards=rewards[start:],
                        terminations=terminations[start:],
                        truncations=truncations[start:],
                        infos=None,
                    )
                )

        return episodes

    def load_metadata(self) -> Metadata:
        with h5py.File(self.path, "r") as h5_file:
            observations = h5_file["observations"]
            actions = h5_file["actions"]
            obs_shape = tuple(int(x) for x in observations.shape[1:])
            act_shape = tuple(int(x) for x in actions.shape[1:])
            count = int(observations.shape[0])

        env_id = D4RL_ENV_IDS.get(self.path.parent.name, "")
        metadata_path = self.path.parent / "metadata.json"
        if not env_id and metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
            env_id = metadata.get("env_id", "")

        return Metadata(
            env_id=env_id,
            obs_shape=obs_shape,
            act_shape=act_shape,
            obs_dim=int(np.prod(obs_shape)) if obs_shape else 1,
            act_dim=int(np.prod(act_shape)) if act_shape else 1,
            count=count,
        )

    def write_buffer(self, path: Path, buffer: Buffer) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rewards = buffer.rewards.detach().cpu().numpy().reshape(-1)
        dones = buffer.dones.detach().cpu().numpy().reshape(-1).astype(np.bool_)
        terminals = dones
        timeouts = np.zeros_like(terminals, dtype=np.bool_)
        with h5py.File(path, "w") as h5_file:
            h5_file.create_dataset("observations", data=buffer.observations.detach().cpu().numpy(), compression="gzip")
            h5_file.create_dataset("next_observations", data=buffer.next_observations.detach().cpu().numpy(), compression="gzip")
            h5_file.create_dataset("actions", data=buffer.actions.detach().cpu().numpy(), compression="gzip")
            h5_file.create_dataset("rewards", data=rewards, compression="gzip")
            h5_file.create_dataset("terminals", data=terminals, compression="gzip")
            h5_file.create_dataset("timeouts", data=timeouts, compression="gzip")


if __name__ == "__main__":
    from ice_offline.dataset._lookup import make_dataset

    dataset = make_dataset("hopper_d4rl_medium", device="cuda")
    loader = dataset.loader
    metadata = loader.load_metadata()
    buffer = loader.load_buffer()
    episodes = loader.load_episodes()

    print(f"env_id={metadata.env_id}")
    print(f"obs_shape={metadata.obs_shape}")
    print(f"act_shape={metadata.act_shape}")
    print(f"count={metadata.count}")
    print(f"buffer_observations_shape={tuple(buffer.observations.shape)}")
    print(f"buffer_actions_shape={tuple(buffer.actions.shape)}")
    print(f"episode_count={len(episodes)}")
