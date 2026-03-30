import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

from ice_offline.env.visualization import BasicUnit, OverlayWrapper
from ice_offline.env.visualization.overlay_engine import RenderLayer
from ice_offline.tools import insert_render_quiet_innermost, now_ns, ns_to_ms


def run_overlay_rgb_array(
    episodes: int = 3,
    max_steps: int = 100,
    *,
    show_gui: bool = True,
    window_name: str = "overlay_rgb_array",
) -> None:
    cv2 = None
    if show_gui:
        try:
            import cv2 as _cv2
        except ImportError as exc:
            raise RuntimeError("show_gui=True requires opencv-python (cv2).") from exc
        cv2 = _cv2
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    env = gym.make("BabyAI-OneRoomS8-v0", render_mode="rgb_array")
    env = insert_render_quiet_innermost(env)
    env = OverlayWrapper(env, units=[BasicUnit()])

    try:
        for episode in range(episodes):
            _, _ = env.reset()
            env.engine.reset_timing()
            print(f"\n=== episode {episode} start ===")

            done = False
            truncated = False
            steps = 0

            while not (done or truncated) and steps < max_steps:
                action = int(np.random.randint(0, 4))

                t0 = now_ns()
                _, reward, done, truncated, _ = env.step(action)
                step_ms = ns_to_ms(now_ns() - t0)

                before_layer_ms = env.engine.get_timing_ms_by_layer()
                t1 = now_ns()
                frame = env.render()
                render_ms = ns_to_ms(now_ns() - t1)
                after_layer_ms = env.engine.get_timing_ms_by_layer()
                if cv2 is not None:
                    # OpenCV expects BGR, while env.render() returns RGB.
                    cv2.imshow(window_name, frame[:, :, ::-1])
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        done = True

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
                    f"done={done} truncated={truncated} step_ms={step_ms:.3f} render_ms={render_ms:.3f} "
                    f"{layer_text}"
                )
                steps += 1

            per_layer = env.engine.get_timing_ms_by_layer()
            print(f"overlay_layer_ms={per_layer}")
            print(f"=== episode {episode} end: steps={steps} done={done} truncated={truncated} ===")
    finally:
        env.close()
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run_overlay_rgb_array()
