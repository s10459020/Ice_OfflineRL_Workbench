import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from collections import defaultdict

from ice_offline.env.visualization import BasicUnit, DistributionUnit, OverlayWrapper
from ice_offline.env.visualization.overlay_engine import RenderLayer
from ice_offline.tools import Timer, insert_render_quiet_innermost


value_table: defaultdict[tuple[bytes, int], dict[int, float]] = defaultdict(dict)


def value_fn(obs, action: int, set_value: float | None = None) -> float:
    key = (obs["image"].tobytes(), obs["direction"])
    if set_value is not None:
        value_table[key][int(action)] = set_value
    return value_table[key].get(int(action), 0.0)


def main(episodes: int = 3) -> None:
    env = gym.make("BabyAI-OneRoomS8-v0", render_mode="human")
    env = insert_render_quiet_innermost(env)
    env = OverlayWrapper(env, units=[BasicUnit(), DistributionUnit(value_fn=value_fn, style="rect")])  # ring12
    try:
        for episode in range(episodes):
            _, _ = env.reset()

            done = False
            truncated = False
            steps = 0

            while not (done or truncated):
                action = int(np.random.randint(0, 4))
                step_ms, step_out = Timer.record("distribution_human.step", lambda: env.step(action))
                obs, _, done, truncated, _ = step_out
                current_value = value_fn(obs, action, set_value=steps)
                render_ms, _ = Timer.record("distribution_human.render", env.render)
                layer_text = " ".join(
                    f"{layer.name.lower()}={Timer.get(f'overlay.layer.{layer.name.lower()}'):.3f}ms"
                    for layer in RenderLayer
                )
                print(
                    f"episode={episode} step={steps:03d} action={action} value={current_value:.1f} "
                    f"step_ms={step_ms:.3f} render_ms={render_ms:.3f} {layer_text}"
                )
                steps += 1
            print(f"episode={episode} steps={steps} done={done} truncated={truncated}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
