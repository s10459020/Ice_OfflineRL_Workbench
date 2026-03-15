from pathlib import Path
from typing import Any
import gymnasium as gym
from agent import Agent

def train(
    env: gym.Env,
    agent: Agent,
    max_steps: int = 100000,
    max_episodes: int | None = None,
    max_episode_steps: int | None = None,
    seed: int | None = None,
    save_model_dir: str | Path | None = None,
    save_model_interval: int | None = None,
    print_interval: int | None = None,
    render_flag: bool = False,
) -> tuple[int, int, dict[str, Any]]:
    """Generic online trainer for env-agent interaction."""
    if save_model_interval is not None and save_model_interval <= 0:
        raise ValueError("save_model_interval must be positive when provided.")
    if print_interval is not None and print_interval <= 0:
        raise ValueError("print_interval must be positive when provided.")
    if max_episodes is not None and max_episodes <= 0:
        return 0, 0, {}
    if max_episode_steps is not None and max_episode_steps <= 0:
        return 0, 0, {}

    save_model_dir = Path(save_model_dir) if save_model_dir is not None else None
    save_model_name = f"{str(env.spec.id)}_{str(agent.agent_name)}"

    steps = 0
    episodes = 0
    episode_steps = 0
    episode_reward = 0.0
    interval_reward_sum = 0.0
    interval_episode_count = 0
    obs, _ = env.reset(seed=seed)
    if render_flag:
        env.render()
    episodes += 1

    info: dict[str, Any] = {}
    if save_model_dir is not None:
        info["save_model_dir"] = save_model_dir

    for _ in range(max_steps):
        if max_episodes is not None and episodes > max_episodes:
            break

        action = int(agent.act(obs))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        if render_flag:
            env.render()
        reward_value = float(reward)
        episode_steps += 1
        steps += 1
        episode_reward += reward_value

        done = bool(terminated or truncated)
        agent.update(obs, action, reward_value, next_obs, done)

        if (
            save_model_dir is not None
            and save_model_interval is not None
            and steps % save_model_interval == 0
        ):
            agent.save(save_model_dir, f"{save_model_name}_step{steps}")

        if done or (max_episode_steps and episode_steps >= max_episode_steps):
            interval_reward_sum += episode_reward
            interval_episode_count += 1
            episodes += 1
            episode_steps = 0
            episode_reward = 0.0
            obs, _ = env.reset(seed=None if seed is None else seed + episodes)
            if render_flag:
                env.render()
        else:
            obs = next_obs

        if print_interval is not None and steps % print_interval == 0:
            interval_avg_score = (
                interval_reward_sum / interval_episode_count
                if interval_episode_count > 0
                else 0.0
            )
            print(
                f"step={steps} episode={episodes} episode_steps={episode_steps} "
                f"interval_avg_score={interval_avg_score:.3f} "
                f"interval_episodes={interval_episode_count}"
            )
            interval_reward_sum = 0.0
            interval_episode_count = 0

    if save_model_dir is not None and steps > 0:
        agent.save(save_model_dir, save_model_name)

    return steps, episodes, info
