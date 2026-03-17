from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from agent import QTableAgent
from strategy import train
from tools import StepPenaltyWrapper
from visualization.minigrid import (
    DistributionWrapper,
    RenderDelayWrapper,
    RenderOverlayWrapper,
    TrailWrapper,
)


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
env = TrailWrapper(
    env,
    clear_on_render=True,
    max_trails=20,
)
env = RenderDelayWrapper(env, fps=3, render_on_done=True)

print(
    "start train | from=scratch | "
    "env=BabyAI-OneRoomS8-v0-fullobs | step_penalty=0.01 | trail=20 | delay_fps=3"
)
try:
    steps, episodes, _ = train(
        env=env,
        agent=agent,
        max_steps=20_000,
        max_episodes=1_000,
        max_episode_steps=200,
        seed=42,
        print_interval=1,
        render_flag=True,
    )
finally:
    env.close()

print(
    f"train_done | steps={steps} | episodes={episodes} | "
    f"q_states={len(agent.q_table)} | epsilon={agent.epsilon:.3f}"
)
