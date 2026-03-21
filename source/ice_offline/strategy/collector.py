import re
from typing import Any, Callable

import gymnasium as gym
import minari

from ice_offline.replay import StateRecordWrapper
from ice_offline.tools import (
    MissionTextWrapper,
    NoJpegImageWrapper,
    insert_render_quiet_innermost,
)


def run(
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
    if re.fullmatch(r".+-v\d+", dataset_id) is None:
        raise ValueError("dataset_id must end with version suffix like '-v0', '-v1', ...")

    env = insert_render_quiet_innermost(env)
    env = MissionTextWrapper(env)
    env = NoJpegImageWrapper(env)
    env = StateRecordWrapper(env)
    env = minari.DataCollector(env, record_infos=True)

    steps = 0
    for episode in range(1, max_episodes + 1):
        obs, _ = env.reset(seed=None if seed is None else seed + episode)
        if render_interval == 1:
            env.render()

        episode_step = 0
        while True:
            action = int(policy(obs))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_step += 1
            steps += 1

            if render_interval is not None and steps % render_interval == 0:
                env.render()

            if print_interval is not None and steps % print_interval == 0:
                print(
                    f"step={steps} episode={episode} episode_step={episode_step} "
                    f"action={action} reward={float(reward):.3f} "
                    f"terminated={terminated} truncated={truncated}"
                )

            if terminated or truncated:
                break
            obs = next_obs

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
            description="Collected by collector",
        )
    finally:
        env.close()

    return steps
