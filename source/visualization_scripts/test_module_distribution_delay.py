from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from agent import QTableAgent
from strategy import test
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


MODEL_PATH = "model/BabyAI-OneRoomS8-v0_QTableAgent.pkl"

agent = QTableAgent.load(MODEL_PATH)
agent.set_encoder(minigrid_q_encoder)
agent.epsilon = 0.05

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
env = RenderDelayWrapper(env, fps=2, render_on_done=True)

print(
    f"start test | loaded_model={MODEL_PATH} | q_states={len(agent.q_table)} | "
    "env=BabyAI-OneRoomS8-v0-fullobs | step_penalty=0.01 | trail=20 | delay_fps=2"
)
try:
    finished_episodes = test(
        env=env,
        max_episodes=2000,
        max_episode_steps=200,
        seed=42,
        policy=lambda obs: int(agent.act(obs, greedy=False)),
        print_flag=True,
    )
finally:
    env.close()

print(
    f"test_done | finished_episodes={finished_episodes} | "
    f"q_states={len(agent.q_table)} | policy_epsilon={agent.epsilon:.3f}"
)
