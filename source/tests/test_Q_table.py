from typing import Any, Callable

import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.
import numpy as np

from agent import QTableAgent


def q_table_state_from_minigrid_observation(obs: Any) -> Any:
    if isinstance(obs, dict) and "image" in obs and "direction" in obs:
        image = np.asarray(obs["image"], dtype=np.uint8)
        direction = int(obs["direction"])
        return direction, image.tobytes()
    raise TypeError("This encoder only supports MiniGrid dict observations with image/direction.")


def train_q_learning(
    env_id: str = "MiniGrid-Empty-5x5-v0",
    episodes: int = 8000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.3,
    max_steps_per_episode: int = 200,
    seed: int = 42,
    log_interval: int = 50,
    max_total_steps: int = 100000,
    state_encoder: Callable[[Any], Any] = q_table_state_from_minigrid_observation,
    allowed_actions: tuple[int, ...] = (0, 1, 2),
) -> QTableAgent:
    env = gym.make(env_id)
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError("Tabular Q-learning requires a discrete action space.")
    if any((a < 0 or a >= int(env.action_space.n)) for a in allowed_actions):
        raise ValueError("allowed_actions contains invalid action ids.")
    if len(allowed_actions) == 0:
        raise ValueError("allowed_actions cannot be empty.")

    agent = QTableAgent(
        n_actions=len(allowed_actions),
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        seed=seed,
    )
    total_steps = 0

    for episode in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + episode)
        state: Any = state_encoder(obs)
        episode_reward = 0.0

        for _ in range(max_steps_per_episode):
            action_idx = agent.act(state)
            env_action = int(allowed_actions[action_idx])
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            next_state = state_encoder(next_obs)
            done = bool(terminated or truncated)
            agent.update(state, action_idx, float(reward), next_state, done)

            state = next_state
            episode_reward += float(reward)
            total_steps += 1
            if done:
                break
            if total_steps >= max_total_steps:
                break

        if episode % log_interval == 0 or episode == 1 or episode == episodes:
            print(
                f"episode={episode:4d}/{episodes} reward={episode_reward:.3f} "
                f"epsilon={agent.epsilon:.3f} states={len(agent.q_table)} total_steps={total_steps}"
            )
        if total_steps >= max_total_steps:
            break

    env.close()
    return agent


def evaluate_q_learning(
    agent: QTableAgent,
    env_id: str = "MiniGrid-Empty-5x5-v0",
    episodes: int = 100,
    max_steps_per_episode: int = 200,
    seed: int = 7,
    state_encoder: Callable[[Any], Any] = q_table_state_from_minigrid_observation,
    allowed_actions: tuple[int, ...] = (0, 1, 2),
) -> tuple[float, float]:
    env = gym.make(env_id)
    if any((a < 0 or a >= int(env.action_space.n)) for a in allowed_actions):
        raise ValueError("allowed_actions contains invalid action ids.")
    if len(allowed_actions) != agent.n_actions:
        raise ValueError("allowed_actions length must match loaded agent.n_actions.")

    success = 0
    rewards: list[float] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        state: Any = state_encoder(obs)
        episode_reward = 0.0

        for _ in range(max_steps_per_episode):
            action_idx = agent.act(state, greedy=True)
            env_action = int(allowed_actions[action_idx])
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            state = state_encoder(next_obs)
            episode_reward += float(reward)

            if terminated:
                success += 1
                break
            if truncated:
                break

        rewards.append(episode_reward)

    env.close()
    success_rate = success / max(1, episodes)
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    return success_rate, avg_reward


if __name__ == "__main__":
    agent = train_q_learning()
    sr, ar = evaluate_q_learning(agent)
    print(
        f"evaluation | success_rate={sr:.3f} | avg_reward={ar:.3f} | "
        f"states={len(agent.q_table)}"
    )
