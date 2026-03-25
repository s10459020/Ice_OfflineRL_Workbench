"""Visualization panel widget for trail and q-table controls."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QGroupBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout


class VisualizationPanel(QGroupBox):
    """Grouped controls for visualization toggles and sliders."""

    def __init__(self) -> None:
        super().__init__("Visualization")
        self._trail_toggle = QCheckBox()
        self._trail_toggle.setFocusPolicy(Qt.NoFocus)
        self._trail_slider = QSlider(Qt.Horizontal)
        self._trail_slider.setRange(0, 100)
        self._trail_slider.setFocusPolicy(Qt.NoFocus)
        self._trail_slider.setStyleSheet(
            """
            QSlider::groove:horizontal { background: #0a1d38; height: 10px; border-radius: 5px; }
            QSlider::handle:horizontal { background: #f11111; width: 26px; border-radius: 13px; margin: -8px 0; }
            """
        )
        self._q_table_toggle = QCheckBox()
        self._q_table_toggle.setFocusPolicy(Qt.NoFocus)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        trail_row = QHBoxLayout()
        trail_row.addWidget(QLabel("Trail"))
        trail_row.addWidget(self._trail_toggle)
        trail_row.addWidget(self._trail_slider, 1)

        q_row = QHBoxLayout()
        q_row.addWidget(QLabel("Q-table"))
        q_row.addWidget(self._q_table_toggle)
        q_row.addStretch(1)

        layout.addLayout(trail_row)
        layout.addLayout(q_row)

    def reset_defaults(self) -> None:
        self._trail_toggle.setChecked(True)
        self._trail_slider.setValue(42)
        self._q_table_toggle.setChecked(False)

