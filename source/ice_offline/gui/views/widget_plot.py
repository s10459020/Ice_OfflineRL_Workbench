import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ice_offline.gui.models.model_action import ActionCurve


class PlotWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._x_name = ""
        self._y_name = ""
        self._curves_1: list[ActionCurve] = []
        self._curves_2: list[ActionCurve] = []
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_axis_names(self, x_name: str, y_name: str) -> None:
        self._x_name = x_name
        self._y_name = y_name
        self.update()

    def set_curves(self, curves_1: list[ActionCurve], curves_2: list[ActionCurve]) -> None:
        self._curves_1 = curves_1
        self._curves_2 = curves_2
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        curves = self._curves_1 or self._curves_2
        if not curves:
            painter.setPen(QColor("#30343b"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No action data")
            return

        left = 72
        top = 24
        right = 48
        bottom = 42
        gap = 14
        panel_count = len(curves)
        plot_width = max(10, self.width() - left - right)
        plot_height = max(10, self.height() - top - bottom - gap * (panel_count - 1))
        panel_height = max(10, plot_height / panel_count)
        painter.setPen(QColor("#30343b"))
        if self._x_name:
            painter.drawText(QRectF(left, self.height() - 28, plot_width, 20), Qt.AlignCenter, self._x_name)
        if self._y_name:
            painter.save()
            painter.translate(18, top + plot_height / 2)
            painter.rotate(-90)
            painter.drawText(QRectF(-plot_height / 2, -18, plot_height, 20), Qt.AlignCenter, self._y_name)
            painter.restore()

        for index in range(panel_count):
            y0 = top + index * (panel_height + gap)
            rect = QRectF(left, y0, plot_width, panel_height)
            curve_1 = self._curves_1[index] if index < len(self._curves_1) else None
            curve_2 = self._curves_2[index] if index < len(self._curves_2) else None
            self._draw_panel(painter, rect, curve_1, curve_2)

    def _draw_panel(self, painter: QPainter, rect: QRectF, curve_1: ActionCurve | None, curve_2: ActionCurve | None) -> None:
        painter.setPen(QPen(QColor("#c9cdd3"), 1.0))
        painter.drawRect(rect)

        painter.setPen(QColor("#30343b"))
        label = curve_1.label if curve_1 is not None else curve_2.label
        painter.drawText(QRectF(rect.left() + 8, rect.top() + 6, 80, 18), label)

        curves = [curve for curve in [curve_1, curve_2] if curve is not None]
        if not curves:
            return

        xs_all = [np.asarray(curve.xs, dtype=np.float32) for curve in curves]
        x_min = float(min(xs.min() for xs in xs_all))
        x_max = float(max(xs.max() for xs in xs_all))
        y_min = 0.0
        y_max = 1.0
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0

        axis_pen = QPen(QColor("#d6d9de"), 1.0)
        painter.setPen(axis_pen)
        if y_min <= 0.0 <= y_max:
            zero_y = self._map_y(0.0, y_min, y_max, rect)
            painter.drawLine(rect.left(), zero_y, rect.right(), zero_y)
        if x_min <= 0.0 <= x_max:
            zero_x = self._map_x(0.0, x_min, x_max, rect)
            painter.drawLine(zero_x, rect.top(), zero_x, rect.bottom())

        if curve_1 is not None:
            self._draw_curve(painter, rect, curve_1, QColor("#2563eb"), x_min, x_max, y_min, y_max)
            painter.setPen(QColor("#30343b"))
            painter.drawText(QRectF(4, rect.top() - 2, rect.left() - 10, 16), Qt.AlignRight | Qt.AlignVCenter, f"{curve_1.source_max:.3g}")
            painter.drawText(QRectF(4, rect.bottom() - 14, rect.left() - 10, 16), Qt.AlignRight | Qt.AlignVCenter, f"{curve_1.source_min:.3g}")
        if curve_2 is not None:
            self._draw_curve(painter, rect, curve_2, QColor("#eab308"), x_min, x_max, y_min, y_max)
            painter.setPen(QColor("#30343b"))
            painter.drawText(QRectF(rect.right() + 10, rect.top() - 2, 38, 16), Qt.AlignLeft | Qt.AlignVCenter, f"{curve_2.source_max:.3g}")
            painter.drawText(QRectF(rect.right() + 10, rect.bottom() - 14, 38, 16), Qt.AlignLeft | Qt.AlignVCenter, f"{curve_2.source_min:.3g}")

        action_value = curve_1.action_value if curve_1 is not None else curve_2.action_value
        action_x = self._map_x(action_value, x_min, x_max, rect)
        painter.setPen(QPen(QColor("#dc2626"), 2.0))
        painter.drawLine(action_x, rect.top(), action_x, rect.bottom())

    def _draw_curve(self, painter: QPainter, rect: QRectF, curve: ActionCurve, color: QColor, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        xs = np.asarray(curve.xs, dtype=np.float32)
        ys = np.asarray(curve.ys, dtype=np.float32)
        if xs.size == 0 or ys.size == 0:
            return

        path = QPainterPath()
        first = QPointF(self._map_x(float(xs[0]), x_min, x_max, rect), self._map_y(float(ys[0]), y_min, y_max, rect))
        path.moveTo(first)
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(self._map_x(float(x), x_min, x_max, rect), self._map_y(float(y), y_min, y_max, rect))

        base_y = self._map_y(0.0, y_min, y_max, rect)
        fill_path = QPainterPath(path)
        fill_path.lineTo(self._map_x(float(xs[-1]), x_min, x_max, rect), base_y)
        fill_path.lineTo(self._map_x(float(xs[0]), x_min, x_max, rect), base_y)
        fill_path.closeSubpath()

        fill_color = QColor(color)
        fill_color.setAlphaF(0.30)
        painter.fillPath(fill_path, fill_color)

        line_color = QColor(color)
        line_color.setAlphaF(0.95)
        painter.setPen(QPen(line_color, 2.0))
        painter.drawPath(path)

    def _map_x(self, value: float, min_value: float, max_value: float, rect: QRectF) -> float:
        ratio = (value - min_value) / (max_value - min_value)
        return rect.left() + ratio * rect.width()

    def _map_y(self, value: float, min_value: float, max_value: float, rect: QRectF) -> float:
        ratio = (value - min_value) / (max_value - min_value)
        return rect.bottom() - ratio * rect.height()
