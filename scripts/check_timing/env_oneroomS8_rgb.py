import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from ice_offline.tools import Timer
from minigrid.wrappers import FullyObsWrapper


def main() -> None:
    env_id = "BabyAI-OneRoomS8-v0"
    episodes = 100

    env = gym.make(env_id, render_mode="rgb_array")
    env = FullyObsWrapper(env)

    total_steps = 0
    total_step_ms = 0.0
    total_render_ms = 0.0
    total_rl_step_ms = 0.0

    try:
        for episode in range(1, episodes + 1):
            env.reset()
            _ = Timer.stopwatch("rl.step")

            while True:
                action = int(np.random.randint(0, env.action_space.n))
                step_ms, step_result = Timer.record("env.step", lambda: env.step(action))
                render_ms, _ = Timer.record("env.render", env.render)
                rl_step_ms = Timer.stopwatch("rl.step")
                _, _, terminated, truncated, _ = step_result

                total_steps += 1
                total_step_ms += step_ms
                total_render_ms += render_ms
                total_rl_step_ms += rl_step_ms

                print(
                    f"episode={episode} "
                    f"env.step_ms={step_ms:.3f} "
                    f"env.render_ms={render_ms:.3f} "
                    f"rl_step_ms={rl_step_ms:.3f}"
                )

                if terminated or truncated:
                    break

        avg_step_ms = total_step_ms / total_steps if total_steps > 0 else 0.0
        avg_render_ms = total_render_ms / total_steps if total_steps > 0 else 0.0
        avg_rl_step_ms = total_rl_step_ms / total_steps if total_steps > 0 else 0.0
        rl_steps_per_sec = 1000.0 / avg_rl_step_ms if avg_rl_step_ms > 0 else float("inf")

        print(
            f"avg_env.step_ms={avg_step_ms:.3f} "
            f"avg_env.render_ms={avg_render_ms:.3f} "
            f"avg_rl_step_ms={avg_rl_step_ms:.3f} "
            f"rl_steps_per_sec={rl_steps_per_sec:.3f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
