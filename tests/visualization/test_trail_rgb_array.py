import minigrid  # noqa: F401
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel

from ice_offline.env.visualization import BasicUnit, OverlayLoader
from ice_offline.env.visualization.overlay_engine import RenderLayer
from ice_offline.env.visualization.unit_trail import TrailUnit
from ice_offline.tools import Timer


def main(dataset: str = "test_collect-v0") -> None:
    qt_app = QApplication.instance() or QApplication([])
    qt_label = QLabel()
    qt_label.resize(480, 480)
    qt_label.setScaledContents(True)
    qt_label.show()

    loader = OverlayLoader(
        dataset=dataset,
        units=[BasicUnit(), TrailUnit(max_trails=8, trail_mode="rollout")],
        render_mode="rgb_array",
    )
    try:
        total_episodes = loader.get_episode_count()
        for ep in range(total_episodes):
            loader.load(ep)
            steps = 0
            t = 1
            while True:
                seek_ms, out = Timer.record("trail_loader_rgb_array.seek", lambda: loader.seek(t))
                _, _, _, terminated, truncated, _ = out

                render_ms, frame = Timer.record("trail_loader_rgb_array.render", loader.render)
                h, w, _ = frame.shape
                rgb = np.ascontiguousarray(frame)
                qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
                qt_label.setPixmap(QPixmap.fromImage(qimg))
                qt_app.processEvents()

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
        loader.close()
        qt_label.close()


if __name__ == "__main__":
    main()
