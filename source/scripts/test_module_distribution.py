from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from agent import QTableAgent
from distribution import DistributionWrapper
from tools import StepPenaltyWrapper
from trainer import train


def minigrid_q_encoder(obs: Any) -> Any:
    image = np.asarray(obs["image"], dtype=np.uint8)
    direction = int(obs["direction"])
    return direction, image.tobytes()


agent = QTableAgent.load("model/BabyAI-OneRoomS8-v0_QTableAgent_step500000.pkl")
agent.set_encoder(minigrid_q_encoder)

env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
env = FullyObsWrapper(env)
env = StepPenaltyWrapper(env, step_penalty=0.01)
env = DistributionWrapper(env)
env.set_q_function(lambda obs, action: agent.q(obs, action))

print(
    "start train | from=model/BabyAI-OneRoomS8-v0_QTableAgent_step500000.pkl | "
    "env=BabyAI-OneRoomS8-v0-fullobs | step_penalty=0.01"
)
try:
    steps, episodes, _ = train(
        env=env,
        agent=agent,
        max_steps=20_000,
        max_episodes=1_000,
        max_episode_steps=200,
        seed=42,
        print_interval=100,
        render_flag=True,
    )
finally:
    env.close()

print(f"train_done | steps={steps} | episodes={episodes} | q_states={len(agent.q_table)}")
