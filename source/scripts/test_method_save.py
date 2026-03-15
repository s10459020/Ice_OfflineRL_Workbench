from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from agent import QTableAgent
from tools import StepPenaltyWrapper
from trainer import train


def q_table_state_from_minigrid_observation(obs: Any) -> Any:
    image = np.asarray(obs["image"], dtype=np.uint8)
    direction = int(obs["direction"])
    return direction, image.tobytes()


env = gym.make("BabyAI-OneRoomS8-v0")
env = FullyObsWrapper(env)
env = StepPenaltyWrapper(env, step_penalty=0.01)
agent = QTableAgent(
    n_actions=4,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.3,
    seed=42,
)
agent.set_encoder(q_table_state_from_minigrid_observation)

try:
    steps, episodes, info = train(
        env=env,
        agent=agent,
        max_steps=1_000_000,
        max_episodes=20_000,
        max_episode_steps=200,
        seed=42,
        save_model_dir="model",
        save_model_interval=50_000,
        print_interval=1000,
    )
finally:
    env.close()

print(
    f"train_done | steps={steps} | episodes={episodes} | "
    f"q_states={len(agent.q_table)} | save_dir={info.get('save_model_dir')}"
)
