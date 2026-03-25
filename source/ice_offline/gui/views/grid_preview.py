"""Grid preview view component for static/mock visualization."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget


class GridPreview(QWidget):
    """Preview panel that draws a static grid sample for frontend validation."""

    # ====================
    # Lifecycle
    # ====================
    def __init__(
        self,
        grid_width: int = 7,
        grid_height: int = 7,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if grid_width <= 0 or grid_height <= 0:
            raise ValueError("grid_width and grid_height must be positive integers.")
        self._grid_width = grid_width
        self._grid_height = grid_height
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ====================
    # Sizing
    # ====================
    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        # Keep the preview aspect ratio aligned with configured grid dimensions.
        return max(1, int(width * self._grid_height / self._grid_width))

    # ====================
    # Painting
    # ====================
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        available = self.rect().adjusted(12, 12, -12, -12)
        target_ratio = self._grid_width / self._grid_height
        available_ratio = available.width() / max(1, available.height())

        # Fit the drawing area inside the widget while preserving the target ratio.
        if available_ratio > target_ratio:
            draw_height = available.height()
            draw_width = int(draw_height * target_ratio)
        else:
            draw_width = available.width()
            draw_height = int(draw_width / target_ratio)

        outer = available.adjusted(
            (available.width() - draw_width) // 2,
            (available.height() - draw_height) // 2,
            -((available.width() - draw_width) // 2),
            -((available.height() - draw_height) // 2),
        )

        painter.fillRect(outer, QColor("#6e6e6e"))

        inner = outer.adjusted(44, 44, -44, -44)
        mid_x = inner.left() + inner.width() // 2

        painter.fillRect(inner.adjusted(0, 0, -(inner.width() // 2), 0), QColor("#525252"))
        painter.fillRect(inner.adjusted(inner.width() // 2, 0, 0, 0), QColor("#070707"))

        pen = QPen(QColor("#666a70"))
        pen.setWidth(2)
        painter.setPen(pen)
        for i in range(0, self._grid_width + 1):
            x = inner.left() + (inner.width() * i) // self._grid_width
            painter.drawLine(x, inner.top(), x, inner.bottom())
        for i in range(0, self._grid_height + 1):
            y = inner.top() + (inner.height() * i) // self._grid_height
            painter.drawLine(inner.left(), y, inner.right(), y)

        painter.fillRect(inner.left(), inner.top(), 24, inner.height(), QColor("#9f9f9f"))
        painter.fillRect(mid_x - 4, inner.top(), 8, inner.height(), QColor("#4f4f4f"))
        painter.setBrush(QColor("#adadad"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(mid_x - 8, inner.top() + 30), 14, 14)

        tri = QPolygonF(
            [
                QPointF(mid_x - 20, inner.center().y()),
                QPointF(mid_x - 52, inner.center().y() + 16),
                QPointF(mid_x - 20, inner.center().y() + 32),
            ]
        )
        painter.setBrush(QColor("#ff5656"))
        painter.drawPolygon(tri)
        painter.end()
