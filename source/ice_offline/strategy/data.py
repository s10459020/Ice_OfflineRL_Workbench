"""Dataset strategy APIs: collect trajectories and view stored dataset episodes."""

import re
from typing import Any, Callable

import gymnasium as gym
import minari

from ice_offline.replay import StateRecordWrapper
from ice_offline.replay.state_inject_wrapper import StateInjectWrapper
from ice_offline.tools import (
    MissionTextWrapper,
    NoJpegImageWrapper,
    insert_render_quiet_innermost,
)

def offline_view(
    dataset: str | Any,
    max_episodes: int | None = None,
    *,
    print_interval: int | None = None,
) -> int:
    """View (replay) a Minari dataset and optionally print progress."""
    # Resolve dataset id/object and validate replay bounds.
    minari_dataset = minari.load_dataset(dataset) if isinstance(dataset, str) else dataset
    total_episodes = minari_dataset.total_episodes

    if total_episodes <= 0:
        return 0
    if max_episodes is not None and total_episodes < max_episodes:
        raise ValueError(f"max_episodes={max_episodes} exceeds total_episodes={total_episodes}.")

    replay_episodes = total_episodes if max_episodes is None else max_episodes
    episode_indices = minari_dataset.episode_indices[:replay_episodes]
    episode_iterable = minari_dataset.iterate_episodes(episode_indices)

    step = 0
    for episode, (trajectory_id, trajectory) in enumerate(zip(episode_indices, episode_iterable), start=1):
        # Replay each stored trajectory and optionally print progress.
        actions = trajectory.actions
        rewards = trajectory.rewards
        terminations = trajectory.terminations
        truncations = trajectory.truncations

        for episode_step in range(len(actions)):
            action = actions[episode_step]
            reward = rewards[episode_step]
            terminated = terminations[episode_step]
            truncated = truncations[episode_step]

            step += 1
            if print_interval is not None and step % print_interval == 0:
                print(
                    f"step={step} episode={episode} trajectory_id={trajectory_id} episode_step={episode_step + 1} "
                    f"action={action} reward={reward:.3f} "
                    f"terminated={terminated} truncated={truncated}"
                )

    return step


def online_view(
    env: gym.Env,
    dataset: str | Any,
    max_episodes: int = 3,
    *,
    seed: int | None = None,
    random_episode: bool = False,
    render_interval: int | None = None,
    print_interval: int | None = None,
) -> int:
    """Replay a dataset through an env-like online loop using StateInjectWrapper."""
    env = insert_render_quiet_innermost(env)
    env = StateInjectWrapper(env, dataset=dataset, random_episode=random_episode)

    steps = 0
    for episode in range(1, max_episodes + 1):
        obs, _ = env.reset(seed=None if seed is None else seed + episode)
        if render_interval == 1:
            env.render()

        episode_step = 0
        while True:
            next_obs, reward, terminated, truncated, info = env.step(None)
            episode_step += 1
            steps += 1

            if render_interval is not None and steps % render_interval == 0:
                env.render()

            if print_interval is not None and steps % print_interval == 0:
                print(
                    f"step={steps} episode={episode} episode_step={episode_step} "
                    f"action={info.get('action')} reward={reward:.3f} "
                    f"terminated={terminated} truncated={truncated}"
                )

            if terminated or truncated:
                break
            obs = next_obs

    env.close()
    return steps


def collect(
    env: gym.Env,
    policy: Callable[[Any], int],
    max_episodes: int = 3,
    dataset_id: str = "collector-v0",
    *,
    seed: int | None = 42,
    render_interval: int | None = None,
    print_interval: int | None = None,
    overwrite: bool = False,
) -> int:
    """Collect trajectories and persist them as a Minari dataset."""
    # Validate dataset id naming convention.
    if re.fullmatch(r".+-v\d+", dataset_id) is None:
        raise ValueError("dataset_id must end with version suffix like '-v0', '-v1', ...")

    # Apply collection wrappers and initialize Minari collector.
    env = insert_render_quiet_innermost(env)
    env = MissionTextWrapper(env)
    env = NoJpegImageWrapper(env)
    env = StateRecordWrapper(env)
    env = minari.DataCollector(env, record_infos=True)

    # Roll out policy trajectories into collector buffers.
    steps = 0
    for episode in range(1, max_episodes + 1):
        obs, _ = env.reset(seed=None if seed is None else seed + episode)
        if render_interval == 1:
            env.render()

        episode_step = 0
        while True:
            action = policy(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_step += 1
            steps += 1

            if render_interval is not None and steps % render_interval == 0:
                env.render()

            if print_interval is not None and steps % print_interval == 0:
                print(
                    f"step={steps} episode={episode} episode_step={episode_step} "
                    f"action={action} reward={reward:.3f} "
                    f"terminated={terminated} truncated={truncated}"
                )

            if terminated or truncated:
                break
            obs = next_obs

    # Create or overwrite target dataset, then always close collector env.
    try:
        if overwrite:
            try:
                minari.delete_dataset(dataset_id)
            except Exception:
                pass

        env.create_dataset(
            dataset_id=dataset_id,
            algorithm_name="custom_policy",
            author="ice_offline",
            author_email="local_test@example.com",
            code_permalink="https://example.com/ice-offline",
            eval_env=env,
            description="Collected by data.collect",
        )
    finally:
        env.close()

    return steps
