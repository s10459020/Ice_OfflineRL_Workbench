from collections.abc import Iterable

import d3rlpy
import numpy as np

from ice_offline.data.minari.loader import MinariLoader
from ice_offline.tools.paths import minari_root


def _to_flatten(minari_episodes: Iterable) -> list[d3rlpy.dataset.Episode]:
    """Converts loaded Minari episodes into d3rlpy Episode list."""
    episodes: list[d3rlpy.dataset.Episode] = []

    for episode in minari_episodes:
        images = np.asarray(episode.observations["image"], dtype=np.float32)
        observations = images.reshape(images.shape[0], -1)
        actions = np.asarray(episode.actions)
        rewards = np.asarray(episode.rewards, dtype=np.float32).reshape(-1, 1)
        terminated = bool(episode.terminations[-1])

        dummy_action = np.zeros_like(actions[-1:])
        dummy_reward = np.zeros_like(rewards[-1:])
        actions = np.concatenate([actions, dummy_action], axis=0)
        rewards = np.concatenate([rewards, dummy_reward], axis=0)

        episodes.append(
            d3rlpy.dataset.Episode(
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
            )
        )

    return episodes


def to_buffer(dataset_id: str, mode: str = "flatten") -> d3rlpy.dataset.ReplayBuffer:
    """Returns train-ready d3rlpy ReplayBuffer converted from Minari dataset id."""
    dataset_path = minari_root() / dataset_id / "data" / "main_data.hdf5"
    minari_dataset = MinariLoader(dataset_path)
    if mode == "flatten":
        episodes = _to_flatten(minari_dataset.iterate_episodes())
    else:
        raise ValueError(f"Unsupported convert mode: {mode}")
    return d3rlpy.dataset.create_infinite_replay_buffer(episodes=episodes)
