from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from ice_offline.agent import QTableAgent
from ice_offline.env.common import insert_render_quiet_innermost
from ice_offline.env.common.step_penalty_wrapper import StepPenaltyWrapper

ENV_ID = "BabyAI-OneRoomS8-v0"
MODEL_ID = f"{ENV_ID}_QTableAgent"
MODEL_STEP = 1000000
MAX_EPISODES = 20


def make_env(env_id: str) -> gym.Env:
    env = gym.make(env_id, render_mode="human")
    env = insert_render_quiet_innermost(env)
    env = FullyObsWrapper(env)
    env = StepPenaltyWrapper(env, step_penalty=0.01)
    return env


def main() -> None:
    agent = QTableAgent.load(
        model_id=MODEL_ID,
        step=MODEL_STEP,
        encoder=lambda obs: np.asarray(obs["image"], dtype=np.uint8).tobytes(),
    )
    env = make_env(ENV_ID)

    total_steps = 0
    try:
        for episode in range(1, MAX_EPISODES + 1):
            obs, _ = env.reset(seed=42 + episode)
            done = False
            truncated = False
            episode_steps = 0

            while not (done or truncated):
                action = int(agent.policy(obs))
                obs, _, done, truncated, _ = env.step(action)
                episode_steps += 1
                total_steps += 1
                print(f"step={total_steps} episode={episode}/{MAX_EPISODES} episode_steps={episode_steps}")
    finally:
        env.close()

    print(f"finished_steps={total_steps}")


if __name__ == "__main__":
    main()
