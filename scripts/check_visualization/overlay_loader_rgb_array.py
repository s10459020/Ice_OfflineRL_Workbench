import minigrid  # noqa: F401
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel

from ice_offline.env.visualization.unit_basic import BasicUnit
from ice_offline.env.visualization.overlay_loader import OverlayLoader
from ice_offline.env.visualization.overlay_engine import RenderLayer
from ice_offline.tools.timing import Timer


# ====================
# Main
# ====================
def main(dataset: str = "test_collect_state-v0") -> None:
    # ---- Qt Display ----
    qt_app = QApplication.instance() or QApplication([])
    qt_label = QLabel()
    qt_label.resize(480, 480)
    qt_label.setScaledContents(True)
    qt_label.show()

    # ---- Dataset Loader ----
    loader = OverlayLoader(dataset=dataset, units=[BasicUnit()], render_mode="rgb_array")
    try:
        # ---- Loop Episodes ----
        total_episodes = loader.get_episode_count()
        for ep in range(total_episodes):
            loader.load(ep)
            steps = 0
            t = 1
            while True:
                # ---- Seek Step ----
                seek_ms, out = Timer.record("overlay_loader_rgb_array.seek", lambda: loader.seek(t))
                _, _, _, terminated, truncated, _ = out

                # ---- Render Frame ----
                render_ms, frame = Timer.record("overlay_loader_rgb_array.render", loader.render)
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
                print(f"episode={ep} seek={t:03d} seek_ms={seek_ms:.3f} render_ms={render_ms:.3f} {layer_text}")
                steps += 1
                if not qt_label.isVisible():
                    return
                if terminated or truncated:
                    break
                t += 1
            print(f"loader episode={ep} steps={steps}")
    finally:
        # ---- Teardown ----
        loader.close()
        qt_label.close()


if __name__ == "__main__":
    main()

