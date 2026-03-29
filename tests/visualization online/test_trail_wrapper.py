import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

from ice_offline.env.visualization import BasicUnit, OverlayWrapper
from ice_offline.env.visualization.overlay_engine import RenderLayer
from ice_offline.env.visualization.unit_trail import TrailUnit
from ice_offline.tools import now_ns, ns_to_ms


def run_trail_wrapper(episodes: int = 3, max_steps: int = 100) -> None:
    env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
    env = OverlayWrapper(env, units=[BasicUnit(), TrailUnit(max_trails=8, trail_mode="rollout")])

    try:
        for episode in range(episodes):
            _, _ = env.reset()
            env.engine.reset_profile()
            print(f"\n=== episode {episode} start ===")

            done = False
            truncated = False
            steps = 0

            while not (done or truncated) and steps < max_steps:
                action = int(np.random.randint(0, 4))

                t0 = now_ns()
                _, reward, done, truncated, _ = env.step(action)
                step_ms = ns_to_ms(now_ns() - t0)
                
                before_layer_ms = env.engine.get_profile_ms_by_layer()
                env.render()
                after_layer_ms = env.engine.get_profile_ms_by_layer()

                step_layer_ms = {
                    layer: after_layer_ms.get(layer, 0.0) - before_layer_ms.get(layer, 0.0)
                    for layer in after_layer_ms
                }
                layer_text = " ".join(
                    f"{RenderLayer(layer).name.lower()}={step_layer_ms[layer]:.3f}ms"
                    for layer in sorted(step_layer_ms)
                )

                print(
                    f"episode={episode} step={steps:03d} "
                    f"action={int(action)} reward={float(reward):.3f} "
                    f"done={done} truncated={truncated} step_ms={step_ms:.3f} "
                    f"{layer_text}"
                )
                steps += 1

            per_layer = env.engine.get_profile_ms_by_layer()
            print(f"overlay_layer_ms={per_layer}")
            print(f"=== episode {episode} end: steps={steps} done={done} truncated={truncated} ===")
    finally:
        env.close()


if __name__ == "__main__":
    run_trail_wrapper()
