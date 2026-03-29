from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from ice_offline.agent import QTableAgent
from ice_offline.strategy import train
from ice_offline.tools import StepPenaltyWrapper
from ice_offline.visualization.minigrid import DistributionWrapper, RenderOverlayWrapper


def minigrid_q_encoder(obs: Any) -> Any:
    image = np.asarray(obs["image"], dtype=np.uint8)
    direction = int(obs["direction"])
    return direction, image.tobytes()


agent = QTableAgent(
    n_actions=4,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.3,
    seed=42,
)
agent.set_encoder(minigrid_q_encoder)

env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
env = FullyObsWrapper(env)
env = StepPenaltyWrapper(env, step_penalty=0.01)
env = RenderOverlayWrapper(env)
env = DistributionWrapper(
    env,
    value_fn=lambda obs, action: agent.q(obs, action),
)

print(
    "start train | from=scratch | "
    "env=BabyAI-OneRoomS8-v0-fullobs | step_penalty=0.01"
)
try:
    steps = train(
        env=env,
        agent=agent,
        max_steps=20_000,
        max_episode_steps=200,
        seed=42,
        print_interval=1,
        render_interval=1,
    )
finally:
    env.close()

print(f"train_done | steps={steps} | q_states={len(agent.q_table)}")
