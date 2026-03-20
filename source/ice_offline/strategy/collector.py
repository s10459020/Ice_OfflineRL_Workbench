from pathlib import Path
from typing import Any, Callable

import gymnasium as gym

from ice_offline.replay import DataTrajectoryManager, StateCollector
from ice_offline.tools import ensure_render_quiet


def collect_dataset(
    env: gym.Env,
    max_episodes: int = 3,
    *,
    seed: int | None = 42,
    policy: Callable[[Any], int] | None = None,
    state_output_path: str | Path | None = None,
    observation_output_path: str | Path | None = None,
    flush_interval: int = 0,
    render_interval: int | None = None,
    print_interval: int | None = None,
) -> int:
    if max_episodes <= 0:
        raise ValueError("max_episodes must be > 0.")
    if state_output_path is None and observation_output_path is None:
        raise ValueError("At least one of state_output_path / observation_output_path must be provided.")
    if flush_interval < 0:
        raise ValueError("flush_interval must be >= 0.")
    if render_interval is not None and render_interval <= 0:
        raise ValueError("render_interval must be > 0 when provided.")
    if print_interval is not None and print_interval <= 0:
        raise ValueError("print_interval must be > 0 when provided.")

    state_collector: StateCollector | None = None
    if state_output_path is not None:
        state_collector = StateCollector(
            env,
            output_path=state_output_path,
            flush_interval=flush_interval,
        )
        env = state_collector

    observation_writer: DataTrajectoryManager | None = None
    if observation_output_path is not None:
        observation_writer = DataTrajectoryManager(
            observation_output_path,
            mode="w",
            flush_interval=flush_interval,
        )

    env = ensure_render_quiet(env)
    if policy is None:
        policy = lambda _obs: env.action_space.sample()

    step = 0
    try:
        for episode in range(1, max_episodes + 1):
            obs, info = env.reset(seed=None if seed is None else seed + episode)
            if render_interval == 1:
                env.render()
            if observation_writer is not None:
                observation_writer.push_observation(obs)

            episode_step = 0
            while True:
                action = int(policy(obs))
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_step += 1
                step += 1

                if observation_writer is not None:
                    observation_writer.push_observation(next_obs)
                if render_interval is not None and step % render_interval == 0:
                    env.render()
                if print_interval is not None and step % print_interval == 0:
                    print(
                        f"step={step} episode={episode} episode_step={episode_step} "
                        f"action={action} reward={float(reward):.3f} "
                        f"terminated={terminated} truncated={truncated}"
                    )

                if terminated or truncated:
                    if observation_writer is not None:
                        observation_writer.end_episode()
                    break
                obs = next_obs
    finally:
        if observation_writer is not None:
            observation_writer.flush()
            observation_writer.close()
        if state_collector is not None:
            state_collector.close_writer()

    return step
