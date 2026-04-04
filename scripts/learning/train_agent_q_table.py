from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from ice_offline.agent import QTableAgent
from ice_offline.env.common.step_penalty_wrapper import StepPenaltyWrapper
from ice_offline.tools import print_stage


MAX_TRAIN_STEPS = 20_000
LOG_EVERY_STEPS = 100
POLICY_EPSILON = 0.3


# ====================
# Init
# ====================
print_stage("Init")
env = gym.make("BabyAI-OneRoomS8-v0")
env = FullyObsWrapper(env)
env = StepPenaltyWrapper(env, step_penalty=0.01)

agent = QTableAgent(
    n_actions=4,
    encoder=lambda obs: (int(obs["direction"]), obs["image"].tobytes()),
    alpha=0.1,
    gamma=0.99,
    epsilon=0.3,
    seed=42,
)


# ====================
# Train
# ====================
print_stage("Train")
total_steps = 0
episode = 0
try:
    while total_steps < MAX_TRAIN_STEPS:
        episode += 1
        obs, _ = env.reset(seed=42 + episode)
        done = False
        truncated = False
        episode_steps = 0

        while not (done or truncated) and total_steps < MAX_TRAIN_STEPS:
            action = int(agent.policy(obs, epsilon=POLICY_EPSILON))
            next_obs, reward, done, truncated, _ = env.step(action)
            agent.update(obs, action, float(reward), next_obs, bool(done or truncated))

            obs = next_obs
            episode_steps += 1
            total_steps += 1

            if total_steps % LOG_EVERY_STEPS == 0:
                print(
                    f"step={total_steps}/{MAX_TRAIN_STEPS} "
                    f"episode={episode} episode_steps={episode_steps} q_states={len(agent.Q)}"
                )

    # ====================
    # Summary
    # ====================
    print_stage("Summary")
    print(f"train_steps={total_steps}")
    print(f"episodes={episode}")
    print(f"q_states={len(agent.Q)}")
finally:
    env.close()



