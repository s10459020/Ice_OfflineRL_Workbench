import os
from pathlib import Path
import re
from typing import Any, Callable

import gymnasium as gym
import minari

from ice_offline.replay import StateRecordWrapper
from ice_offline.tools import insert_render_quiet_innermost


def collect_dataset(
    env: gym.Env,
    max_episodes: int = 3,
    *,
    seed: int | None = 42,
    policy: Callable[[Any], int] | None = None,
    dataset_id: str | None = None,
    flush_interval: int = 0,
    render_interval: int | None = None,
    print_interval: int | None = None,
    record_infos: bool = True,
    overwrite: bool = False,
    algorithm_name: str = "random",
    author: str = "ice_offline",
    author_email: str = "local_test@example.com",
    code_permalink: str = "https://example.com/ice-offline",
    description: str = "Collected by ice_offline.strategy.collector.collect_dataset",
) -> int:
    if max_episodes <= 0:
        raise ValueError("max_episodes must be > 0.")
    if render_interval is not None and render_interval <= 0:
        raise ValueError("render_interval must be > 0 when provided.")
    if print_interval is not None and print_interval <= 0:
        raise ValueError("print_interval must be > 0 when provided.")
    if flush_interval < 0:
        raise ValueError("flush_interval must be >= 0.")

    if dataset_id is None:
        raise ValueError("dataset_id is required.")
    if re.fullmatch(r".+-v\d+", dataset_id) is None:
        raise ValueError("dataset_id must end with version suffix like '-v0', '-v1', ...")

    # Fixed Minari root for this project.
    minari_root = Path("tmps") / "ice_datasets"
    minari_root.mkdir(parents=True, exist_ok=True)
    os.environ["MINARI_DATASETS_PATH"] = str(minari_root.resolve())

    env = insert_render_quiet_innermost(env)
    if record_infos:
        env = StateRecordWrapper(env)

    collector = minari.DataCollector(env, record_infos=record_infos)
    if policy is None:
        policy = lambda _obs: collector.action_space.sample()

    steps = 0
    for episode in range(1, max_episodes + 1):
        obs, _ = collector.reset(seed=None if seed is None else seed + episode)
        if render_interval == 1:
            collector.render()

        episode_step = 0
        while True:
            action = int(policy(obs))
            next_obs, reward, terminated, truncated, _ = collector.step(action)
            episode_step += 1
            steps += 1

            if render_interval is not None and steps % render_interval == 0:
                collector.render()
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

        collector.create_dataset(
            dataset_id=dataset_id,
            algorithm_name=algorithm_name,
            author=author,
            author_email=author_email,
            code_permalink=code_permalink,
            eval_env=env,
            description=description,
        )
    finally:
        collector.close()

    return steps
