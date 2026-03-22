"""Offline strategy APIs: train from dataset and validation against dataset behavior."""

from typing import Any, Callable

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
    total_episodes = int(dataset_obj.total_episodes)
    if total_episodes <= 0:
        return 0

    if max_episodes is None:
        train_episodes = total_episodes
    else:
        train_episodes = min(int(max_episodes), total_episodes)
        
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
        act_seq = np.asarray(trajectory.actions)
        rew_seq = np.asarray(trajectory.rewards, dtype=np.float64)
        term_seq = np.asarray(trajectory.terminations, dtype=bool)
        trunc_seq = np.asarray(trajectory.truncations, dtype=bool)

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


def validation(
    dataset: str | Any,
    max_episodes: int | None = None,
    *,
    policy: Callable[[Any], int],
    seed: int | None = None,
    sample_flag: bool = False,
    print_interval: int | None = None,
) -> dict[str, float | int]:
    """Validation against behavior actions and rewards in a Minari dataset."""
    # Resolve dataset id/object and bound evaluation episodes.
    dataset_obj = minari.load_dataset(dataset) if isinstance(dataset, str) else dataset
    total_episodes = int(dataset_obj.total_episodes)
    if total_episodes <= 0:
        return {
            "episodes": 0,
            "steps": 0,
            "total_reward": 0.0,
            "mean_episode_reward": 0.0,
            "action_match_rate": 0.0,
        }

    if max_episodes is None:
        eval_episodes = total_episodes
    else:
        eval_episodes = min(int(max_episodes), total_episodes)
        
    trajectory_indices = _select_trajectory_indices(
        dataset_obj=dataset_obj,
        replay_episodes=eval_episodes,
        sample_flag=sample_flag,
        seed=seed,
    )

    total_steps = 0
    total_reward = 0.0
    matched_actions = 0

    for episode_index, trajectory_index in enumerate(trajectory_indices):
        # Compare policy output with behavior actions trajectory by trajectory.
        trajectory = dataset_obj[trajectory_index]
        obs_seq = _materialize_obs_seq(trajectory.observations)
        act_seq = np.asarray(trajectory.actions)
        rew_seq = np.asarray(trajectory.rewards, dtype=np.float64)

        for episode_step in range(len(act_seq)):
            # Accumulate action agreement and reward statistics per step.
            obs = obs_seq[episode_step]
            behavior_action = int(act_seq[episode_step])
            predicted_action = int(policy(obs))
            if predicted_action == behavior_action:
                matched_actions += 1

            reward = float(rew_seq[episode_step])
            total_reward += reward
            total_steps += 1

            if print_interval is not None and total_steps % print_interval == 0:
                print(
                    f"step={total_steps} episode={episode_index} episode_step={episode_step + 1} "
                    f"action={predicted_action} behavior={behavior_action} reward={reward:.3f}"
                )

    mean_episode_reward = total_reward / eval_episodes if eval_episodes > 0 else 0.0
    action_match_rate = matched_actions / total_steps if total_steps > 0 else 0.0
    return {
        "episodes": eval_episodes,
        "steps": total_steps,
        "total_reward": float(total_reward),
        "mean_episode_reward": float(mean_episode_reward),
        "action_match_rate": float(action_match_rate),
    }


def _materialize_obs_seq(observations: Any) -> list[Any]:
    # Convert batched observation container into per-step observations once.
    if isinstance(observations, dict):
        observation_arrays = {k: np.asarray(v) for k, v in observations.items()}
        step_count = len(next(iter(observation_arrays.values()))) if observation_arrays else 0
        return [
            {k: observation_arrays[k][episode_step] for k in observation_arrays}
            for episode_step in range(step_count)
        ]
    return list(np.asarray(observations))


def _select_trajectory_indices(
    dataset_obj: Any,
    replay_episodes: int,
    sample_flag: bool,
    seed: int | None,
) -> list[int]:
    # Build deterministic front-slice or random sampled trajectory indices.
    candidate_indices = np.asarray(dataset_obj.episode_indices, dtype=np.int64)
    if sample_flag:
        random_generator = np.random.default_rng(seed)
        return random_generator.choice(candidate_indices, size=replay_episodes, replace=False).tolist()
    return candidate_indices[:replay_episodes].tolist()
