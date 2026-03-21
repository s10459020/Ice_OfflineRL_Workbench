from typing import Any, Callable

import numpy as np

import minari


def run(
    dataset: str | Any,
    max_episodes: int | None = None,
    *,
    policy: Callable[[Any], int] | None = None,
    print_interval: int | None = None,
) -> dict[str, float | int]:
    """Offline test flow using a Minari dataset.

    Args:
        dataset: Minari dataset id or loaded Minari dataset object.
        max_episodes: Max episodes to evaluate. `None` means all episodes.
        policy: Offline policy function `policy(observation) -> action`.
                If None, uses behavior actions from dataset.
        print_interval: Print every N steps when provided.
    """
    if print_interval is not None and print_interval <= 0:
        raise ValueError("print_interval must be > 0 when provided.")

    ds = minari.load_dataset(dataset) if isinstance(dataset, str) else dataset
    total_episodes = int(ds.total_episodes)
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
        if max_episodes <= 0:
            raise ValueError("max_episodes must be > 0 when provided.")
        eval_episodes = min(int(max_episodes), total_episodes)

    total_steps = 0
    total_reward = 0.0
    matched_actions = 0

    for episode_index in range(eval_episodes):
        episode = ds[episode_index]
        obs_seq = episode.observations
        act_seq = np.asarray(episode.actions)
        rew_seq = np.asarray(episode.rewards, dtype=np.float64)

        for t in range(len(act_seq)):
            obs_t = _obs_at(obs_seq, t)
            behavior_action = int(act_seq[t])
            predicted_action = behavior_action if policy is None else int(policy(obs_t))
            if predicted_action == behavior_action:
                matched_actions += 1

            reward = float(rew_seq[t])
            total_reward += reward
            total_steps += 1

            if print_interval is not None and total_steps % print_interval == 0:
                print(
                    f"step={total_steps} episode={episode_index} t={t} "
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


def _obs_at(observations: Any, index: int) -> Any:
    if isinstance(observations, dict):
        return {k: np.asarray(v)[index] for k, v in observations.items()}
    return np.asarray(observations)[index]

