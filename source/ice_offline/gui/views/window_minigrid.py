"""Main window for a frontend-only dataset viewer MVP mockup."""


from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ice_offline.gui.presenters.replay_minigrid import ViewerPresenter
from ice_offline.gui.services.minari_dataset_service import MinariDatasetService
from ice_offline.gui.views.grid_render_panel import GridRenderPanel
from ice_offline.gui.views.visualization_panel import VisualizationPanel


class MainWindow(QMainWindow):
    """Frontend-only mockup for layout validation and basic MVP interaction."""

    # ====================
    # init & reset
    # ====================
    def __init__(self, presenter: ViewerPresenter, dataset_service: MinariDatasetService) -> None:
        super().__init__()
        self.setWindowTitle("Dataset GUI Mockup")
        self.resize(980, 640)
        self.setFocusPolicy(Qt.StrongFocus)
        self._presenter = presenter
        self._dataset_service = dataset_service
        self._list: QListWidget | None = None
        self._slider: QSlider | None = None
        self._grid_view: GridRenderPanel | None = None
        self._setting_area: VisualizationPanel | None = None
        self._build_ui()
        self._bind_presenter()
        self._initialize_view_state()

    def _initialize_view_state(self) -> None:
        # Fill episode list first, then apply default UI state.
        self._list.blockSignals(True)
        self._list.clear()
        for label in self._presenter.episode_labels():
            self._list.addItem(QListWidgetItem(label))
        self._list.blockSignals(False)
        self._reset_ui_state()

    def _reset_ui_state(self) -> None:
        self._list.setCurrentRow(0)
        self._setting_area.reset_defaults()
        self._dataset_service.set_distribution_enabled(self._setting_area.is_q_table_enabled())
        self._on_episode_selected(self._list.currentRow())



    # ====================
    # UI
    # ====================
    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(20)
        self.setCentralWidget(root)

        # ========================================
        # ||             Root Layout            ||
        # || +--------------------------------+ ||
        # || | Content Layout                 | ||
        # || +--------------------------------+ ||
        # || +--------------------------------+ ||
        # || | Step Slider                    | ||
        # || +--------------------------------+ ||
        # ========================================
        content_layout = QHBoxLayout()
        content_layout.setSpacing(24)
        root_layout.addLayout(content_layout, 1)
        
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setFixedHeight(30)
        self._slider.setFocusPolicy(Qt.NoFocus)
        self._slider.setStyleSheet(
            """
            QSlider::groove:horizontal { background: #071427; height: 10px; border-radius: 5px; }
            QSlider::handle:horizontal { background: #6b9fd6; width: 30px; border-radius: 15px; margin: -10px 0; }
            """
        )
        root_layout.addWidget(self._slider)

        # +------------------------------------------+
        # |              Content Layout              |
        # |  +---------------+   +-----------------+ |
        # |  | Grid Render   |   | Control Layout  | |
        # |  +---------------+   +-----------------+ |
        # +------------------------------------------+
        self._grid_view = GridRenderPanel()
        content_layout.addWidget(self._grid_view, 3)

        control_layout = QVBoxLayout()
        control_layout.setSpacing(18)
        control_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addLayout(control_layout, 2)

        # +--------------------------+
        # |   Right: Control Layout  |
        # |  +-------------------+   |
        # |  | Select List       |   |
        # |  +-------------------+   |
        # |  +-------------------+   |
        # |  | Plugin Area       |   |
        # |  +-------------------+   |
        # +--------------------------+
        self._list = QListWidget()
        self._list.setFocusPolicy(Qt.NoFocus)
        group_box = QGroupBox("Episodes")
        layout = QVBoxLayout(group_box)
        layout.addWidget(self._list)
        control_layout.addWidget(group_box)
        
        self._setting_area = VisualizationPanel()
        control_layout.addWidget(self._setting_area)
        control_layout.addStretch(1)

    # ====================
    # Widget Events
    # ====================
    def _bind_presenter(self) -> None:
        self._list.currentRowChanged.connect(self._presenter._on_list_changed)
        self._slider.valueChanged.connect(self._presenter._on_slider_changed)
        self._setting_area.on_changed(self._presenter._on_setting_changed)

    def _on_list_changed(self, episode_index: int) -> None:
        min_step, max_step, step = self._presenter.on_episode_selected(episode_index)
        self._slider.blockSignals(True)
        self._slider.setRange(min_step, max_step)
        self._slider.setValue(step)
        self._slider.blockSignals(False)
        self._sync_grid_view()
        
    def _set_render(self) -> None:
        frame = self._presenter.render()
        self._grid_view.set_frame(frame)

    # ====================
    # Qt Native Events
    # ====================
    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()

        if key == Qt.Key_Up:
            row = max(0, self._list.currentRow() - 1)
            self._list.setCurrentRow(row)
            event.accept()
            return

        if key == Qt.Key_Down:
            row = min(self._list.count() - 1, self._list.currentRow() + 1)
            self._list.setCurrentRow(row)
            event.accept()
            return

        if key == Qt.Key_Left:
            self._slider.setValue(self._slider.value() - 1)
            event.accept()
            return

        if key == Qt.Key_Right:
            self._slider.setValue(self._slider.value() + 1)
            event.accept()
            return

        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._dataset_service.close()
        super().closeEvent(event)
