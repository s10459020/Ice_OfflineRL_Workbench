import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.
import numpy as np
from minigrid.wrappers import FullyObsWrapper
import time


env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
env = FullyObsWrapper(env)

try:
    episodes = 0
    total_steps = 0

    for episode in range(10):
        env.reset()
        episodes += 1

        for step in range(10):
            action = int(np.random.randint(0, 4))
            t0 = time.perf_counter_ns()
            _, reward, terminated, truncated, _ = env.step(action)
            step_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

            t1 = time.perf_counter_ns()
            frame = env.render()
            render_ms = (time.perf_counter_ns() - t1) / 1_000_000.0

            total_steps += 1

            print(
                f"episode={episode} step={step:03d} action={int(action)} "
                f"reward={float(reward):.3f} step_ms={step_ms:.3f} render_ms={render_ms:.3f} "
                f"frame_shape={getattr(frame, 'shape', None)} "
                f"terminated={terminated} truncated={truncated}"
            )

            if terminated or truncated:
                break

    print(f"episodes={episodes} total_steps={total_steps}")
finally:
    env.close()
