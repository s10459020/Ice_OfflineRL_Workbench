from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from ice_offline.agent import QTableAgent
from ice_offline.tools import insert_render_quiet_innermost


def main(
    model_path: str,
    env_id: str,
    max_episodes: int = 20,
) -> None:
    agent = QTableAgent.load(
        model_path,
        encoder=lambda obs: (int(obs["direction"]), obs["image"].tobytes()),
    )
    env = gym.make(env_id, render_mode="human")
    env = insert_render_quiet_innermost(env)
    env = FullyObsWrapper(env)

    try:
        total_steps = 0
        for episode in range(1, max_episodes + 1):
            obs, _ = env.reset(seed=42 + episode)
            done = False
            truncated = False
            episode_steps = 0

            while not (done or truncated):
                action = int(agent.policy(obs))
                obs, _, done, truncated, _ = env.step(action)
                episode_steps += 1
                total_steps += 1
                print(
                    f"step={total_steps} "
                    f"episode={episode}/{max_episodes} "
                    f"episode_steps={episode_steps}"
                )
    finally:
        env.close()

    print(f"finished_steps={total_steps}")


if __name__ == "__main__":
    env_id = "BabyAI-OneRoomS8-v0"
    model_path = "data/BabyAI-OneRoomS8-v0_QTableAgent_step1000000.pkl"
    max_episodes = 20

    main(
        model_path=model_path,
        env_id=env_id,
        max_episodes=max_episodes,
    )
