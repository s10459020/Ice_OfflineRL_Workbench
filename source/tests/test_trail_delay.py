import sys
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401  # Ensure MiniGrid environments are registered.

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scheduler.minigrid import RenderDelayWrapper, TrailDelayWrapper


def run_with_trail_delay(steps: int = 6000, fps: int = 5) -> None:
    env = gym.make("MiniGrid-FourRooms-v0", render_mode="human", max_steps=6000)
    env = RenderDelayWrapper(env, fps=fps, render_on_done=True)
    env = TrailDelayWrapper(env)

    obs, _ = env.reset()
    print(f"reset | obs_keys={list(obs.keys())} | render_fps={fps}")

    env.render()
    for step in range(1, steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        print(
            f"step={step} action={action} reward={float(reward):.3f} "
            f"terminated={terminated} truncated={truncated} trails={len(env.trails)}"
        )

        if terminated or truncated:
            obs, _ = env.reset()
            print(f"reset | obs_keys={list(obs.keys())}")

    env.close()


if __name__ == "__main__":
    run_with_trail_delay(steps=60000, fps=5)
