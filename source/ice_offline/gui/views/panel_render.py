"""Grid render panel for displaying env.render() RGB frames."""


import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget


class GridRenderPanel(QWidget):
    """Render-only panel that draws the current RGB frame."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: np.ndarray | None = None

    def set_frame(self, frame: np.ndarray) -> None:
        self._frame = frame
        self.update()

    # ====================
    # Qt Native Events
    # ====================
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self._frame is not None:
            frame = np.ascontiguousarray(self._frame)
            height, width, channels = frame.shape
            image = QImage(frame.data, width, height, channels * width, QImage.Format_RGB888)
            painter.drawPixmap(self.rect(), QPixmap.fromImage(image))
        painter.end()
