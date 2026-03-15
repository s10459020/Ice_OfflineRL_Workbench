from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from agent import QTableAgent
from tester import test


def q_table_state_from_minigrid_observation(obs: Any) -> Any:
    image = np.asarray(obs["image"], dtype=np.uint8)
    direction = int(obs["direction"])
    return direction, image.tobytes()


agent = QTableAgent.load("model/BabyAI-OneRoomS8-v0_QTableAgent.pkl")
agent.set_encoder(q_table_state_from_minigrid_observation)
agent.epsilon = 0.05

print("loaded model: model/BabyAI-OneRoomS8-v0_QTableAgent.pkl")
print(f"q_states={len(agent.q_table)} | env=BabyAI-OneRoomS8-v0-fullobs | policy_epsilon={agent.epsilon:.3f}")

env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
env = FullyObsWrapper(env)

try:
    finished_episodes = test(
        env=env,
        max_episodes=20,
        max_episode_steps=20,
        seed=None,
        policy=lambda obs: int((0, 1, 2, 3)[agent.act(obs, greedy=False)]),
        print_flag=True,
    )
finally:
    env.close()

print(f"finished_episodes={finished_episodes}")
