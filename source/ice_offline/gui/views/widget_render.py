import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget


class RenderWidget(QWidget):
    """Render-only panel that draws the current RGB frame."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: np.ndarray | None = None

    # ====================
    # Public API
    # ====================
    def set_frame(self, frame: np.ndarray) -> None:
        self._frame = np.ascontiguousarray(frame)
        self.update()

    # ====================
    # Qt Native Events
    # ====================
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self._frame is not None:
            height, width, channels = self._frame.shape
            image = QImage(self._frame.data, width, height, channels * width, QImage.Format_RGB888)
            painter.drawPixmap(self.rect(), QPixmap.fromImage(image))
        painter.end()
