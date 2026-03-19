from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from ice_offline.agent import QTableAgent
from ice_offline.strategy import test, train
from ice_offline.tools import StepPenaltyWrapper


def q_table_state_from_minigrid_observation(obs: Any) -> Any:
    image = np.asarray(obs["image"], dtype=np.uint8)
    direction = int(obs["direction"])
    return direction, image.tobytes()


############################################################
#  Train
############################################################
train_env = gym.make("BabyAI-OneRoomS8-v0")
train_env = FullyObsWrapper(train_env)
train_env = StepPenaltyWrapper(train_env, step_penalty=0.01)
agent = QTableAgent(
    n_actions=4,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.3,
    seed=42,
)
agent.set_encoder(q_table_state_from_minigrid_observation)

try:
    train_steps = train(
        env=train_env,
        agent=agent,
        max_steps=20_000,
        max_episode_steps=200,
        seed=42,
        print_interval=1_000,
    )
finally:
    train_env.close()


############################################################
#  Test
############################################################
test_env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
test_env = FullyObsWrapper(test_env)
test_env = StepPenaltyWrapper(test_env, step_penalty=0.01)
try:
    finished_steps = test(
        env=test_env,
        max_episodes=20,
        seed=7,
        policy=lambda obs: int(agent.act(obs, greedy=True)),
    )
finally:
    test_env.close()

print(
    f"train_steps={train_steps} | test_steps={finished_steps} | q_states={len(agent.q_table)}"
)
