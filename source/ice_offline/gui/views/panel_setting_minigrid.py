"""Visualization panel widget for q-table controls."""


from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout


class VisualizationPanel(QGroupBox):
    """Grouped controls for visualization toggles."""

    def __init__(self) -> None:
        super().__init__("Visualization")
        self._q_table_toggle = QCheckBox()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        q_row = QHBoxLayout()
        self._q_table_toggle.setFocusPolicy(Qt.NoFocus)
        q_row.addWidget(self._q_table_toggle)
        q_row.addWidget(QLabel("Q-table"))
        q_row.addStretch(1)

        layout.addLayout(q_row)

    def reset_defaults(self) -> None:
        self._q_table_toggle.setChecked(False)

    def is_q_table_enabled(self) -> bool:
        return self._q_table_toggle.isChecked()

    def bind_q_table_toggled(self, callback: Callable[[bool], None]) -> None:
        self._q_table_toggle.toggled.connect(callback)
