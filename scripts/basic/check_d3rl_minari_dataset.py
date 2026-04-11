from typing import Any

import d3rlpy
import minari
import numpy as np

# mute logging
d3rlpy.logging.LOG.info = lambda *args, **kwargs: None

DATASET_ID = "minigrid-fourrooms-v0"
DEVICE = "cuda:0"
RUN_FIT = False


def _extract_observations(episode: Any) -> np.ndarray:
    """Pick a numeric observation stream from Minari episode."""
    obs = episode.observations
    if isinstance(obs, dict):
        if "image" in obs:
            return np.asarray(obs["image"], dtype=np.float32)
        first_key = next(iter(obs.keys()))
        return np.asarray(obs[first_key], dtype=np.float32)
    return np.asarray(obs, dtype=np.float32)


def to_d3rl_replay_buffer(dataset_id: str) -> d3rlpy.dataset.ReplayBuffer:
    minari_dataset = minari.load_dataset(dataset_id)
    episodes: list[d3rlpy.dataset.Episode] = []

    for episode in minari_dataset.iterate_episodes():
        observations = _extract_observations(episode)
        actions = np.asarray(episode.actions)
        rewards = np.asarray(episode.rewards, dtype=np.float32).reshape(-1, 1)
        terminations = np.asarray(episode.terminations, dtype=np.float32)
        truncations = np.asarray(episode.truncations, dtype=np.float32)

        # d3rlpy Episode expects one flag for whole episode.
        # True means environment termination, False means timeout/truncation.
        terminated = bool(terminations[-1]) if len(terminations) > 0 else False
        clipped = bool((terminations[-1] or truncations[-1])) if len(terminations) > 0 else False

        if not clipped:
            # Minari episodes should end with termination or truncation.
            continue

        episodes.append(
            d3rlpy.dataset.Episode(
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
            )
        )

    return d3rlpy.dataset.create_infinite_replay_buffer(episodes=episodes)


def main() -> None:
    dataset = to_d3rl_replay_buffer(DATASET_ID)

    print("dataset_type:", type(dataset))
    print("episode_count:", len(dataset.episodes))
    print("transition_count:", dataset.transition_count)
    print("action_space:", dataset.dataset_info.action_space)
    print("action_size:", dataset.dataset_info.action_size)

    for i, episode in enumerate(dataset.episodes[:3]):
        print(
            f"episode={i} size={episode.size()} "
            f"terminated={episode.terminated} "
            f"transition_count={episode.transition_count}"
        )

    if RUN_FIT:
        algo = d3rlpy.algos.DQNConfig(batch_size=32).create(device=DEVICE)
        algo.fit(dataset, n_steps=max(100, dataset.transition_count), n_steps_per_epoch=100)
        print("fit finished")


if __name__ == "__main__":
    main()
