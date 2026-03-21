from typing import Any, Callable

import gymnasium as gym

from ice_offline.tools import insert_render_quiet_innermost

def run(
    env: gym.Env,
    max_episodes: int = 100,
    *,
    seed: int | None = None,
    policy: Callable[[Any], int],
    render_interval: int | None = None,
    print_interval: int | None = None,
) -> int:
    env = insert_render_quiet_innermost(env)

    step = 0
    for episode in range(1, max_episodes + 1):
        episode_seed = None if seed is None else seed + episode
        obs, _ = env.reset(seed=episode_seed)

        if render_interval == 1:
            env.render()

        episode_step = 0
        while True:
            action = int(policy(obs))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_step += 1
            step += 1

            if render_interval is not None and step % render_interval == 0:
                env.render()

            if print_interval is not None and step % print_interval == 0:
                print(
                    f"step={step} episode={episode} episode_step={episode_step} "
                    f"action={action} reward={float(reward):.3f} "
                    f"terminated={terminated} truncated={truncated}"
                )

            if terminated or truncated:
                break

            obs = next_obs

    return step
