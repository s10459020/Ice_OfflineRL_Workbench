from typing import Any, Callable

import gymnasium as gym

from ice_offline.tools import ensure_render_quiet

def test(
    env: gym.Env,
    max_episodes: int = 100,
    *,
    seed: int | None = None,
    policy: Callable[[Any], int] | None = None,
    render_interval: int | None = None,
    print_interval: int | None = None,
) -> int:
    if max_episodes <= 0:
        raise ValueError("max_episodes must be > 0.")
    if render_interval is not None and render_interval <= 0:
        raise ValueError("render_interval must be > 0 when provided.")
    if print_interval is not None and print_interval <= 0:
        raise ValueError("print_interval must be > 0 when provided.")

    env = ensure_render_quiet(env)
    if policy is None:
        policy = lambda _obs: env.action_space.sample()

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
