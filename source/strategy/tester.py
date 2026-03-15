from typing import Any, Callable

import gymnasium as gym


Policy = Callable[[Any], int]


def test(
    env: gym.Env,
    max_episodes: int = 100,
    max_episode_steps: int = 200,
    seed: int | None = None,
    policy: Policy | None = None,
    print_flag: bool = False,
    render_flag: bool = False,
) -> int:
    if max_episodes <= 0:
        return 0
    if max_episode_steps <= 0:
        return 0
    episodes = 0
    steps = 0
    for episode in range(1, max_episodes + 1):
        obs, _ = env.reset(seed=None if seed is None else seed + episode)
        if render_flag:
            env.render()
        episodes += 1
        for episode_step in range(1, max_episode_steps + 1):
            action = int(env.action_space.sample()) if policy is None else int(policy(obs))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if render_flag:
                env.render()
            steps += 1

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
