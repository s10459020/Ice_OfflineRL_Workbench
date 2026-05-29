from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGroupBox, QLabel, QHBoxLayout, QSpinBox, QVBoxLayout, QWidget


class SettingWidget(QWidget):
    changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._group = QGroupBox("Setting")
        self._label = QLabel("Step Jump")

        self._spin = QSpinBox()
        self._spin.setRange(1, 1000)
        self._spin.setValue(1)
        self._spin.setFocusPolicy(Qt.NoFocus)

        row = QHBoxLayout()
        row.addWidget(self._label)
        row.addWidget(self._spin)

        group_layout = QVBoxLayout(self._group)
        group_layout.addLayout(row)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._group)

        self._spin.valueChanged.connect(self.changed.emit)

    # ====================
    # Public API
    # ====================
    def set_title(self, title: str) -> None:
        self._group.setTitle(title)

    def set_value(self, value: int) -> None:
        self._spin.blockSignals(True)
        self._spin.setValue(value)
        self._spin.blockSignals(False)

    def value(self) -> int:
        return int(self._spin.value())
