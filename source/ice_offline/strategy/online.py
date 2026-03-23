"""Online strategy APIs: environment-interaction train and test flows."""

from pathlib import Path
from typing import Any, Callable

import gymnasium as gym

from ice_offline.agent import Agent
from ice_offline.tools import insert_render_quiet_innermost


def test(
    env: gym.Env,
    policy: Callable[[Any], int],
    max_episodes: int = 100,
    *,
    seed: int | None = None,
    render_interval: int | None = None,
    print_interval: int | None = None,
) -> int:
    """Run policy evaluation episodes directly on the environment."""
    # Prepare wrappers and initialize counters.
    env = insert_render_quiet_innermost(env)

    step = 0
    for episode in range(1, max_episodes + 1):
        # Reset one episode with deterministic seed offset when provided.
        episode_seed = None if seed is None else seed + episode
        obs, _ = env.reset(seed=episode_seed)
        
        if print_interval == 1:
            print(f"reset episode={episode} seed={episode_seed}")

        if render_interval == 1:
            env.render()

        episode_step = 0
        while True:
            # Run policy action and advance one environment step.
            action = policy(obs)
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

    # Return total executed steps across episodes.
    return step


def train(
    env: gym.Env,
    agent: Agent,
    max_steps: int = 100000,
    *,
    seed: int | None = None,
    save_model_dir: str | Path | None = None,
    save_model_interval: int | None = None,
    render_interval: int | None = None,
    print_interval: int | None = None,
) -> int:
    """Generic online trainer for env-agent interaction."""
    # Prepare wrappers, output path, and save naming.
    env = insert_render_quiet_innermost(env)
    env_id = env.spec.id

    save_model_path = Path(save_model_dir) if save_model_dir is not None else None
    save_model_name = f"{env_id}_{agent.agent_name}"

    step = 0
    episode = 0
    while step < max_steps:
        # Start a new episode and reset with optional deterministic seed.
        episode += 1
        episode_seed = None if seed is None else seed + episode
        obs, _ = env.reset(seed=episode_seed)
        if render_interval == 1:
            env.render()

        episode_step = 0
        episode_reward = 0.0
        while True:
            # Interact with environment and apply one TD-style agent update.
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            reward_value = reward
            done = terminated or truncated

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

            obs = next_obs

    # Save final model snapshot after training loop completes.
    if save_model_path is not None and step > 0:
        agent.save(save_model_path, save_model_name)

    return step
