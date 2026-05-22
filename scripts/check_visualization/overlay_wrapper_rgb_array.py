import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel

from ice_offline.env.visualization.unit_basic import BasicUnit`nfrom ice_offline.env.visualization.overlay_wrapper import OverlayWrapper
from ice_offline.env.visualization.overlay_engine import RenderLayer
from ice_offline.env.common.render_quiet_wrapper import insert_render_quiet_innermost
from ice_offline.tools.timing import Timer


def main(episodes: int = 10,) -> None:
    # ---- Qt Display ----
    qt_app = QApplication.instance() or QApplication([])
    qt_label = QLabel()
    qt_label.resize(480, 480)
    qt_label.setScaledContents(True)
    qt_label.show()

    # ---- Environment ----
    env = gym.make("BabyAI-OneRoomS8-v0", render_mode="rgb_array")
    env = insert_render_quiet_innermost(env)
    env = OverlayWrapper(env, units=[BasicUnit()])

    try:
        for episode in range(episodes):
            _, _ = env.reset()
            print(f"episode={episode} start")

            done = False
            truncated = False
            steps = 0

            while not (done or truncated):
                # ---- Step Action ----
                action = int(np.random.randint(0, 4))
                _, _, done, truncated, _ = env.step(action)

                # ---- Render Frame ----
                frame = env.render()
                h, w, _ = frame.shape
                rgb = np.ascontiguousarray(frame)
                qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)

                # ---- Update Qt ----
                qt_label.setPixmap(QPixmap.fromImage(qimg))
                qt_app.processEvents()

                # ---- Log / Exit ----
                layer_text = " ".join(
                    f"{layer.name.lower()}={Timer.get(f'overlay.layer.{layer.name.lower()}'):.3f}ms"
                    for layer in RenderLayer
                )
                print(f"episode={episode} step={steps:03d} {layer_text}")
                if not qt_label.isVisible():
                    done = True
                steps += 1
            print(f"episode={episode} end steps={steps} done={done} truncated={truncated}")

    finally:
        env.close()
        qt_label.close()


if __name__ == "__main__":
    main()

