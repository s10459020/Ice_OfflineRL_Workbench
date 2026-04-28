from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from ice_offline.agent import QTableAgent
from ice_offline.env.common import insert_render_quiet_innermost
from ice_offline.env.common.step_penalty_wrapper import StepPenaltyWrapper
from ice_offline.tools import print_stage

ENV_ID = "BabyAI-OneRoomS8-v0"
MODEL_ID = f"{ENV_ID}_QTableAgent"

MAX_STEPS = 1_000_000
CHECKPOINT_EVERY_STEPS = 50_000
LOG_EVERY_STEPS = 100
POLICY_EPSILON = 0.3


def make_env(env_id: str) -> gym.Env:
    env = gym.make(env_id)
    env = insert_render_quiet_innermost(env)
    env = FullyObsWrapper(env)
    env = StepPenaltyWrapper(env, step_penalty=0.01)
    return env


def main() -> None:
    print_stage("Init")
    env = make_env(ENV_ID)
    agent = QTableAgent(
        n_actions=4,
        encoder=lambda obs: np.asarray(obs["image"], dtype=np.uint8).tobytes(),
        alpha=0.1,
        gamma=0.99,
        seed=42,
    )

    print_stage("Train")
    total_steps = 0
    episode = 0
    try:
        while total_steps < MAX_STEPS:
            episode += 1
            obs, _ = env.reset(seed=42 + episode)
            done = False
            truncated = False

            while not (done or truncated) and total_steps < MAX_STEPS:
                action = int(agent.policy(obs, epsilon=POLICY_EPSILON))
                next_obs, reward, done, truncated, _ = env.step(action)
                agent.update(obs, action, float(reward), next_obs, bool(done or truncated))
                obs = next_obs
                total_steps += 1

                if total_steps % CHECKPOINT_EVERY_STEPS == 0:
                    model_path = agent.save(MODEL_ID, total_steps)
                    print(f"checkpoint={model_path}")

                if total_steps % LOG_EVERY_STEPS == 0:
                    print(f"step={total_steps}/{MAX_STEPS} episode={episode} q_states={len(agent.Q)}")

        print_stage("Summary")
        print(f"train_steps={total_steps}")
        print(f"episodes={episode}")
        print(f"q_states={len(agent.Q)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
