from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QSlider, QVBoxLayout, QWidget

class SliderWidget(QWidget):
    changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._slider = QSlider(Qt.Horizontal, self)
        self._slider.setFocusPolicy(Qt.NoFocus)
        self._slider.setFixedHeight(30)
        self._slider.setStyleSheet(
            """
            QSlider::groove:horizontal { background: #071427; height: 10px; border-radius: 5px; }
            QSlider::handle:horizontal { background: #6b9fd6; width: 30px; border-radius: 15px; margin: -10px 0; }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._slider)

        self._slider.valueChanged.connect(self.changed.emit)

    # ====================
    # Public API   
    # ====================
    def set_range(self, min_value: int, max_value: int, value: int | None = None) -> None:
        self._slider.blockSignals(True)
        self._slider.setRange(min_value, max_value)
        if value is not None:
            self._slider.setValue(value)
        self._slider.blockSignals(False)

    def set_value(self, value: int) -> None:
        self._slider.setValue(value)

    def value(self) -> int:
        return self._slider.value()