from pathlib import Path

import gymnasium as gym

from ice_offline.agent import Agent
from ice_offline.tools import insert_render_quiet_innermost


def run(
    env: gym.Env,
    agent: Agent,
    max_steps: int = 100000,
    *,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    save_model_dir: str | Path | None = None,
    save_model_interval: int | None = None,
    render_interval: int | None = None,
    print_interval: int | None = None,
) -> int:
    """Generic online trainer for env-agent interaction."""
    env = insert_render_quiet_innermost(env)
    save_model_path = Path(save_model_dir) if save_model_dir is not None else None

    env_id = str(env.spec.id) if getattr(env, "spec", None) is not None else "env"
    save_model_name = f"{env_id}_{str(agent.agent_name)}"

    step = 0
    episode = 0
    while step < max_steps:
        episode += 1
        episode_seed = None if seed is None else seed + episode
        obs, _ = env.reset(seed=episode_seed)
        if render_interval == 1:
            env.render()

        episode_step = 0
        episode_reward = 0.0
        while True:
            action = int(agent.act(obs))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            reward_value = float(reward)
            done = bool(terminated or truncated)

            episode_step += 1
            step += 1
            episode_reward += reward_value
            agent.update(obs, action, reward_value, next_obs, done)

            if save_model_path is not None and save_model_interval is not None and step % save_model_interval == 0:
                agent.save(save_model_path, f"{save_model_name}_step{step}")
            if render_interval is not None and step % render_interval == 0:
                env.render()
            if print_interval is not None and step % print_interval == 0:
                print(
                    f"step={step} episode={episode} episode_step={episode_step} "
                    f"action={action} reward={reward_value:.3f} "
                    f"episode_reward={episode_reward:.3f} "
                    f"terminated={terminated} truncated={truncated}"
                )

            if step >= max_steps:
                break
            if done:
                break
            if max_episode_steps is not None and episode_step >= max_episode_steps:
                break

            obs = next_obs

    if save_model_path is not None and step > 0:
        agent.save(save_model_path, save_model_name)

    return step
