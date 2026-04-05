from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from ice_offline.agent import QTableAgent
from ice_offline.tools import print_stage


def main(
    env_id: str,
    model_dir: str,
    model_name_template: str,
    max_train_steps: int = 1_000_000,
    checkpoint_every_steps: int = 50_000,
    log_every_steps: int = 100,
    epsilon: float = 0.3,
) -> None:
    print_stage("Init")
    env = gym.make(env_id)
    env = FullyObsWrapper(env)

    agent = QTableAgent(
        n_actions=4,
        encoder=lambda obs: (int(obs["direction"]), obs["image"].tobytes()),
        alpha=0.1,
        gamma=0.99,
        seed=42,
    )

    print_stage("Train")
    total_steps = 0
    episode = 0
    try:
        while total_steps < max_train_steps:
            episode += 1
            obs, _ = env.reset(seed=42 + episode)
            done = False
            truncated = False
            episode_steps = 0

            while not (done or truncated) and total_steps < max_train_steps:
                action = agent.policy(obs, epsilon=epsilon)
                next_obs, reward, done, truncated, _ = env.step(action)
                agent.update(obs, action, float(reward), next_obs, bool(done or truncated))

                obs = next_obs
                episode_steps += 1
                total_steps += 1

                if total_steps % checkpoint_every_steps == 0:
                    model_name = model_name_template.format(steps=total_steps)
                    model_path = agent.save(model_dir=model_dir, model_name=model_name)
                    print(f"checkpoint={model_path}")

                if total_steps % log_every_steps == 0:
                    print(
                        f"step={total_steps}/{max_train_steps} "
                        f"episode={episode} episode_steps={episode_steps}"
                    )

        print_stage("Summary")
        print(f"train_steps={total_steps}")
        print(f"episodes={episode}")
        print(f"q_states={len(agent.Q)}")
        final_model_name = model_name_template.format(steps=total_steps)
        print(f"final_model={model_dir}/{final_model_name}")
    finally:
        env.close()


if __name__ == "__main__":
    env_id = "BabyAI-OneRoomS8-v0"
    model_dir = "model"
    model_name_template = f"{env_id}_QTableAgent_step{{steps}}.pkl"

    main(
        env_id=env_id,
        model_dir=model_dir,
        model_name_template=model_name_template,
    )
