import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

from ice_offline.env.visualization.unit_basic import BasicUnit
from ice_offline.env.visualization.overlay_wrapper import OverlayWrapper
from ice_offline.env.visualization.overlay_engine import RenderLayer
from ice_offline.env.common.render_quiet_wrapper import insert_render_quiet_innermost
from ice_offline.tools.timing import Timer


def main(episodes: int = 3) -> None:
    env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
    env = insert_render_quiet_innermost(env)
    env = OverlayWrapper(env, units=[BasicUnit()])
    try:
        for episode in range(episodes):
            _, _ = env.reset()

            done = False
            truncated = False
            steps = 0

            while not (done or truncated):
                action = int(np.random.randint(0, 4))
                step_ms, step_out = Timer.record("overlay_wrapper_human.step", lambda: env.step(action))
                _, _, done, truncated, _ = step_out
                render_ms, _ = Timer.record("overlay_wrapper_human.render", env.render)
                layer_text = " ".join(
                    f"{layer.name.lower()}={Timer.get(f'overlay.layer.{layer.name.lower()}'):.3f}ms"
                    for layer in RenderLayer
                )
                print(
                    f"episode={episode} step={steps:03d} step_ms={step_ms:.3f} render_ms={render_ms:.3f} {layer_text}"
                )
                steps += 1
            print(f"episode={episode} steps={steps} done={done} truncated={truncated}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

