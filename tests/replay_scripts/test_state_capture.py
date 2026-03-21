import time

import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.
from minigrid.wrappers import FullyObsWrapper

from ice_offline.replay import StateRecordWrapper


env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
env = FullyObsWrapper(env)
env = StateRecordWrapper(env)

print("start | env=BabyAI-OneRoomS8-v0-fullobs | state_capture=on | render=human")
try:
    episodes = 3
    max_episode_steps = 20
    fps = 6
    frame_delay = 1.0 / fps
    captured_count = 0
    first_states = []
    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=42 + ep)
        _ = (obs, info)
        reset_state = info["state"]
        captured_count += 1
        if len(first_states) < 10:
            first_states.append(reset_state)
        print(
            f"episode={ep} step=0 "
            f"pos={reset_state.agent_pos} dir={reset_state.agent_dir} "
            f"mission='{reset_state.mission}' carrying={reset_state.carrying} "
            f"grid_shape={reset_state.grid.shape}"
        )
        env.render()
        time.sleep(frame_delay)
        for step in range(1, max_episode_steps + 1):
            action = env.action_space.sample()
            _, _, terminated, truncated, info = env.step(action)
            state = info["state"]
            captured_count += 1
            if len(first_states) < 10:
                first_states.append(state)
            carrying = state.carrying if state.carrying is not None else "None"
            print(
                f"episode={ep} step={step} "
                f"pos={state.agent_pos} dir={state.agent_dir} "
                f"mission='{state.mission}' carrying={carrying} "
                f"grid_shape={state.grid.shape}"
            )
            env.render()
            time.sleep(frame_delay)
            if terminated or truncated:
                break
finally:
    env.close()

print(f"captured_states={captured_count}")
for state in first_states:
    print(
        f"pos={state.agent_pos} dir={state.agent_dir} "
        f"mission='{state.mission}' carrying={state.carrying} "
        f"grid_shape={state.grid.shape}"
    )
