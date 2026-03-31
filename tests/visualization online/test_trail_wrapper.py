import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

from ice_offline.env.visualization import BasicUnit, OverlayWrapper
from ice_offline.env.visualization.overlay_engine import RenderLayer
from ice_offline.env.visualization.unit_trail import TrailUnit
from ice_offline.tools import Timer
import time


def run_trail_wrapper(episodes: int = 3, max_steps: int = 100) -> None:
    env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
    env = OverlayWrapper(env, units=[BasicUnit(), TrailUnit(max_trails=8, trail_mode="rollout")])

    try:
        for episode in range(episodes):
            _, _ = env.reset()
            print(f"\n=== episode {episode} start ===")

            done = False
            truncated = False
            steps = 0

            while not (done or truncated) and steps < max_steps:
                action = int(np.random.randint(0, 4))

                t0 = time.perf_counter_ns()
                _, reward, done, truncated, _ = env.step(action)
                step_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                env.render()
                layer_text = " ".join(
                    f"{layer.name.lower()}={Timer.get(f'overlay.layer.{layer.name.lower()}'):.3f}ms"
                    for layer in RenderLayer
                )

                print(
                    f"episode={episode} step={steps:03d} "
                    f"action={int(action)} reward={float(reward):.3f} "
                    f"done={done} truncated={truncated} step_ms={step_ms:.3f} "
                    f"{layer_text}"
                )
                steps += 1

            print(f"=== episode {episode} end: steps={steps} done={done} truncated={truncated} ===")
    finally:
        env.close()


if __name__ == "__main__":
    run_trail_wrapper()

