from typing import Any, Callable

import gymnasium as gym

from ice_offline.tools import ensure_render_quiet

Policy = Callable[[Any], int]


def test(
    env: gym.Env,
    policy: Policy | None = None,
    max_episodes: int = 100,
    seed: int | None = None,
    print_flag: bool = False,
    render_flag: bool = False,
) -> int:
    if max_episodes <= 0:
        raise ValueError("max_episodes must be > 0.")
    if policy is None:
        policy = lambda _: env.action_space.sample()

    episodes = 0
    steps = 0

    env = ensure_render_quiet(env)
    for episode in range(1, max_episodes + 1):
        obs, _ = env.reset(seed=None if seed is None else seed + episode)
        if render_flag:
            env.render()
        episodes += 1

        episode_step = 0
        while True:
            episode_step += 1
            action = int(policy(obs))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1

            if render_flag:
                env.render()

            if print_flag:
                print(
                    f"step={steps} episode={episode} episode_step={episode_step} "
                    f"action={action} reward={float(reward):.3f} "
                    f"terminated={terminated} truncated={truncated}"
                )

            if terminated or truncated:
                break

            obs = next_obs

    return episodes
