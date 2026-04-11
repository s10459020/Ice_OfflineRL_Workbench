from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from ice_offline.agent import QTableAgent
from ice_offline.strategy import test


def q_table_state_from_minigrid_observation(obs: Any) -> Any:
    image = np.asarray(obs["image"], dtype=np.uint8)
    direction = int(obs["direction"])
    return direction, image.tobytes()


agent = QTableAgent.load(
    "data/BabyAI-OneRoomS8-v0_QTableAgent.pkl",
    encoder=q_table_state_from_minigrid_observation,
)
policy_epsilon = 0.05

print("loaded model: data/BabyAI-OneRoomS8-v0_QTableAgent.pkl")
print(f"q_states={len(agent.Q)} | env=BabyAI-OneRoomS8-v0-fullobs | policy_epsilon={policy_epsilon:.3f}")

env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
env = FullyObsWrapper(env)

try:
    finished_steps = test(
        env=env,
        max_episodes=20,
        seed=None,
        policy=lambda obs: int((0, 1, 2, 3)[agent.policy(obs, epsilon=policy_epsilon)]),
        print_interval=1,
    )
finally:
    env.close()

print(f"finished_steps={finished_steps}")






