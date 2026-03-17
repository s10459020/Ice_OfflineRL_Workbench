from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gymnasium as gym

from replay import StateDatasetWriter

Policy = Callable[[Any], int]


def collect_state_dataset(
    env: gym.Env,
    output_path: str | Path,
    max_episodes: int = 100,
    max_episode_steps: int | None = None,
    seed: int | None = None,
    policy: Policy | None = None,
    write_interval: int = 0,
    print_flag: bool = False,
) -> dict[str, Any]:
    if max_episodes <= 0:
        return {"episodes": 0, "steps": 0, "path": str(Path(output_path))}
    if max_episode_steps is not None and max_episode_steps <= 0:
        return {"episodes": 0, "steps": 0, "path": str(Path(output_path))}

    writer = StateDatasetWriter(output_path=output_path, write_interval=write_interval)
    env = writer.wrap_env(env)
    if policy is None:
        policy = lambda _obs: env.action_space.sample()

    total_steps = 0
    collected_episodes = 0

    try:
        for episode in range(1, max_episodes + 1):
            obs, info = env.reset(seed=None if seed is None else seed + episode)
            writer.on_reset(info)

            episode_step = 0
            while True:
                episode_step += 1
                action = int(policy(obs))
                next_obs, reward, terminated, truncated, info = env.step(action)
                writer.on_step(action, reward, terminated, truncated, info)
                total_steps += 1

                done = bool(terminated or truncated)
                forced_cutoff = max_episode_steps is not None and episode_step >= max_episode_steps
                if print_flag:
                    print(
                        f"step={total_steps} episode={episode} episode_step={episode_step} "
                        f"action={action} reward={float(reward):.3f} "
                        f"terminated={terminated} truncated={truncated}"
                    )
                if done:
                    if print_flag:
                        print(
                            f"episode_end episode={episode} episode_steps={episode_step} "
                            "reason=terminated_or_truncated"
                        )
                    break
                if forced_cutoff:
                    writer.end_episode()
                    if print_flag:
                        print(
                            f"episode_end episode={episode} episode_steps={episode_step} "
                            "reason=max_episode_steps"
                        )
                    break
                obs = next_obs

            collected_episodes += 1
    finally:
        writer.close()

    return {
        "episodes": collected_episodes,
        "steps": total_steps,
        "path": str(Path(output_path)),
    }
