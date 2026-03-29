"""Offline strategy APIs: train from dataset."""

from typing import Any

import numpy as np

import minari


def train(
    dataset: str | Any,
    agent: Any,
    max_episodes: int | None = None,
    *,
    seed: int | None = None,
    sample_flag: bool = False,
    print_interval: int | None = None,
) -> int:
    """Train an agent from offline transitions in a Minari dataset."""
    # Resolve dataset id/object and bound episode count.
    dataset_obj = minari.load_dataset(dataset) if isinstance(dataset, str) else dataset
    total_episodes = dataset_obj.total_episodes
    if total_episodes <= 0:
        return 0

    if max_episodes is None:
        train_episodes = total_episodes
    else:
        train_episodes = min(max_episodes, total_episodes)
        
    trajectory_indices = _select_trajectory_indices(
        dataset_obj=dataset_obj,
        replay_episodes=train_episodes,
        sample_flag=sample_flag,
        seed=seed,
    )

    total_steps = 0
    for episode_index, trajectory_index in enumerate(trajectory_indices):
        # Load one episode arrays for transition-wise agent updates.
        trajectory = dataset_obj[trajectory_index]
        obs_seq = _materialize_obs_seq(trajectory.observations)
        act_seq = trajectory.actions
        rew_seq = trajectory.rewards
        term_seq = trajectory.terminations
        trunc_seq = trajectory.truncations

        for episode_step in range(len(act_seq)):
            # Convert dataset row into (s, a, r, s', done) and update agent.
            obs = obs_seq[episode_step]
            next_obs = obs_seq[episode_step + 1]
            action = act_seq[episode_step]
            reward = rew_seq[episode_step]
            done = term_seq[episode_step] or trunc_seq[episode_step]
            agent.update(obs, action, reward, next_obs, done)

            total_steps += 1
            if print_interval is not None and total_steps % print_interval == 0:
                print(
                    f"step={total_steps} episode={episode_index} episode_step={episode_step + 1} "
                    f"trajectory_id={trajectory_index} action={action} reward={reward:.3f} done={done}"
                )

    return total_steps


def _materialize_obs_seq(observations: Any) -> list[Any]:
    # Convert batched observation container into per-step observations once.
    if isinstance(observations, dict):
        step_count = len(next(iter(observations.values()))) if observations else 0
        return [
            {k: observations[k][episode_step] for k in observations}
            for episode_step in range(step_count)
        ]
    return list(observations)


def _select_trajectory_indices(
    dataset_obj: Any,
    replay_episodes: int,
    sample_flag: bool,
    seed: int | None,
) -> list[int]:
    # Build deterministic front-slice or random sampled trajectory indices.
    candidate_indices = list(dataset_obj.episode_indices)
    if sample_flag:
        random_generator = np.random.default_rng(seed)
        return random_generator.choice(candidate_indices, size=replay_episodes, replace=False).tolist()
    return candidate_indices[:replay_episodes]
