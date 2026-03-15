from typing import Any, Callable

import gymnasium as gym

from tools import RenderQuiteWrapper

Policy = Callable[[Any], int]


def _ensure_innermost_render_quite(env: gym.Env) -> None:
    """Insert RenderQuiteWrapper closest to the base env in wrapper chain."""
    if not isinstance(env, gym.Wrapper):
        raise TypeError("test() expects a gym.Wrapper env so RenderQuiteWrapper can be inserted in-place.")

    current = env
    while isinstance(current, gym.Wrapper) and isinstance(current.env, gym.Wrapper):
        current = current.env

    current.env = RenderQuiteWrapper(current.env)


def test(
    env: gym.Env,
    max_episodes: int = 100,
    max_episode_steps: int = 200,
    seed: int | None = None,
    policy: Policy | None = None,
    print_flag: bool = False,
) -> int:
    if max_episodes <= 0 or max_episode_steps <= 0:
        return 0
    if policy is None:
        policy = lambda _obs: env.action_space.sample()

    episodes = 0
    global_step = 0

    _ensure_innermost_render_quite(env)
    for episode in range(1, max_episodes + 1):
        obs, _ = env.reset(seed=None if seed is None else seed + episode)
        env.render()
        episodes += 1

        for episode_step in range(1, max_episode_steps + 1):
            action = int(policy(obs))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            global_step += 1

            if print_flag:
                print(
                    f"step={global_step} episode={episode} episode_step={episode_step} "
                    f"action={action} reward={float(reward):.3f} "
                    f"terminated={terminated} truncated={truncated}"
                )

            if terminated or truncated:
                break

            obs = next_obs

    return episodes
