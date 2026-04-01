"""Visualization panel widget for trail and q-table controls."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QGroupBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout


class VisualizationPanel(QGroupBox):
    """Grouped controls for visualization toggles and sliders."""

    def __init__(self) -> None:
        super().__init__("Visualization")
        self._trail_toggle = QCheckBox()
        self._trail_slider = QSlider(Qt.Horizontal)
        self._q_table_toggle = QCheckBox()
        self._build_ui()

    def _build_ui(self) -> None:
        # ==========================
        # || Visualization Panel ||
        # || +------------------+ ||
        # || | Trail Row        | ||
        # || +------------------+ ||
        # || | Q-table Row      | ||
        # || +------------------+ ||
        # ==========================
        layout = QVBoxLayout(self)

        # +----------------------------------------+
        # |              Trail Row                 |
        # | ==================  ================== |
        # | || Trail Toggle ||  || Trail Slider || |
        # | ==================  ================== |
        # +----------------------------------------+
        trail_row = QHBoxLayout()
        self._trail_toggle.setFocusPolicy(Qt.NoFocus)
        trail_row.addWidget(self._trail_toggle)
        trail_row.addWidget(QLabel("Trail"))
        
        self._trail_slider.setRange(0, 100)
        self._trail_slider.setFocusPolicy(Qt.NoFocus)
        self._trail_slider.setStyleSheet(
            """
            QSlider::groove:horizontal { background: #0a1d38; height: 10px; border-radius: 5px; }
            QSlider::handle:horizontal { background: #f11111; width: 26px; border-radius: 13px; margin: -8px 0; }
            """
        )
        trail_row.addWidget(self._trail_slider, 1)

        # +--------------------------------------+
        # |             Q-table Row              |
        # | =================                    |
        # | || Q-table     ||                    |
        # | =================                    |
        # +--------------------------------------+
        q_row = QHBoxLayout()
        self._q_table_toggle.setFocusPolicy(Qt.NoFocus)
        q_row.addWidget(self._q_table_toggle)
        q_row.addWidget(QLabel("Q-table"))
        q_row.addStretch(1)
        
        
        layout.addLayout(trail_row)
        layout.addLayout(q_row)

    def reset_defaults(self) -> None:
        self._trail_toggle.setChecked(True)
        self._trail_slider.setValue(42)
        self._q_table_toggle.setChecked(False)

    def is_trail_enabled(self) -> bool:
        return self._trail_toggle.isChecked()

    def bind_trail_toggled(self, callback: Callable[[bool], None]) -> None:
        self._trail_toggle.toggled.connect(callback)

    def is_q_table_enabled(self) -> bool:
        return self._q_table_toggle.isChecked()

    def bind_q_table_toggled(self, callback: Callable[[bool], None]) -> None:
        self._q_table_toggle.toggled.connect(callback)
