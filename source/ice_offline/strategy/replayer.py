from typing import Any

import numpy as np
import minari


def run(
    dataset: str | Any,
    max_episodes: int | None = None,
    *,
    seed: int | None = None,
    sample_flag: bool = False,
    print_interval: int | None = None,
) -> int:
    """Replay a Minari dataset and optionally print progress."""
    # ---- Load Dataset ----
    minari_dataset = minari.load_dataset(dataset) if isinstance(dataset, str) else dataset
    total_episodes = minari_dataset.total_episodes

    # ---- Validate Episode Count ----
    if total_episodes <= 0:
        return 0
    if total_episodes < max_episodes:
        raise ValueError(f"max_episodes={max_episodes} exceeds total_episodes={total_episodes}.")

    # ---- Build Episode Indices ----
    replay_episodes = total_episodes if max_episodes is None else max_episodes
    candidate_indices = np.asarray(minari_dataset.episode_indices, dtype=np.int64)

    if sample_flag:
        random_generator = np.random.default_rng(seed)
        episode_indices = random_generator.choice(candidate_indices, size=replay_episodes, replace=False).tolist()
    else:
        episode_indices = candidate_indices[:replay_episodes].tolist()

    # ---- Iterate Dataset Episodes ----
    episode_iterable = minari_dataset.iterate_episodes(episode_indices)

    # ---- Replay Transitions ----
    step = 0
    for episode, (trajectory_id, trajectory) in enumerate(zip(episode_indices, episode_iterable), start=1):
        actions = trajectory.actions
        rewards = trajectory.rewards
        terminations = trajectory.terminations
        truncations = trajectory.truncations

        for episode_step in range(len(actions)):
            action = int(actions[episode_step])
            reward = float(rewards[episode_step])
            terminated = bool(terminations[episode_step])
            truncated = bool(truncations[episode_step])

            step += 1
            if print_interval is not None and step % print_interval == 0:
                print(
                    f"step={step} episode={episode} trajectory_id={trajectory_id} episode_step={episode_step + 1} "
                    f"action={action} reward={reward:.3f} "
                    f"terminated={terminated} truncated={truncated}"
                )

    # ---- Return Metrics ----
    return step
