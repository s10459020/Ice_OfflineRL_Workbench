from __future__ import annotations

import gymnasium as gym
import numpy as np

from ice_offline.agent import PolicyGradientAgent
from ice_offline.tools import print_stage

ENV_ID = "InvertedPendulum-v5"
MAX_EPISODES = 100
LOG_EVERY_EPISODES = 1


def make_env(env_id: str) -> gym.Env:
    return gym.make(env_id)


def main() -> None:
    print_stage("Init")
    env = make_env(ENV_ID)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    agent = PolicyGradientAgent(
        action_dim=action_dim,
        obs_dim=obs_dim,
    )

    print_stage("Train")
    total_steps = 0
    try:
        for episode in range(1, MAX_EPISODES + 1):
            obs, _ = env.reset(seed=42 + episode)
            done = False
            truncated = False
            episode_reward = 0.0
            episode_steps = 0

            while not (done or truncated):
                action = agent.act(obs)
                next_obs, reward, done, truncated, _ = env.step(action)
                agent.record_step(obs, action, reward)

                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

            agent.update_episode()

            if episode % LOG_EVERY_EPISODES == 0:
                print(
                    f"episode={episode}/{MAX_EPISODES} "
                    f"episode_steps={episode_steps} "
                    f"episode_reward={episode_reward:.3f} "
                    f"total_steps={total_steps}"
                )

        print_stage("Summary")
        print(f"episodes={MAX_EPISODES}")
        print(f"total_steps={total_steps}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
