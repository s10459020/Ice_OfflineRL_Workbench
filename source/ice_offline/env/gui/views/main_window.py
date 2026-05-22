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

from ice_offline.env.gui.presenters.viewer_presenter import ViewerPresenter
from ice_offline.env.gui.services.minari_dataset_service import MinariDatasetService
from ice_offline.env.gui.views.grid_render_panel import GridRenderPanel
from ice_offline.env.gui.views.visualization_panel import VisualizationPanel


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
        self._episode_list: QListWidget | None = None
        self._step_slider: QSlider | None = None
        self._grid_view: GridRenderPanel | None = None
        self._visualization_panel: VisualizationPanel | None = None
        self._build_ui()
        self._bind_presenter()
        self._initialize_view_state()

    def _initialize_view_state(self) -> None:
        # Fill episode list first, then apply default UI state.
        self._episode_list.blockSignals(True)
        self._episode_list.clear()
        for label in self._presenter.episode_labels():
            self._episode_list.addItem(QListWidgetItem(label))
        self._episode_list.blockSignals(False)
        self._reset_ui_state()

    def _reset_ui_state(self) -> None:
        self._episode_list.setCurrentRow(0)
        self._visualization_panel.reset_defaults()
        self._dataset_service.set_trail_enabled(self._visualization_panel.is_trail_enabled())
        self._dataset_service.set_distribution_enabled(self._visualization_panel.is_q_table_enabled())
        self._on_episode_selected(self._episode_list.currentRow())



    # ====================
    # UI
    # ====================
    def _build_ui(self) -> None:
        # ========================================
        # ||             Root Layout            ||
        # || +--------------------------------+ ||
        # || | Content Row                    | ||
        # || +--------------------------------+ ||
        # || +--------------------------------+ ||
        # || | Step Slider                    | ||
        # || +--------------------------------+ ||
        # ========================================
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(20)

        # ========================================
        # ||             Content Row            ||
        # || +----------------+ +-------------+ ||
        # || | Grid Render    | | Right Pane  | ||
        # || +----------------+ +-------------+ ||
        # ========================================
        content_row = QHBoxLayout()
        content_row.setSpacing(24)
        root_layout.addLayout(content_row, 1)

        # +--------------------------------------+
        # |              Content Row             |
        # |  ==================  +-------------+ |
        # |  || Grid Render  ||  | Right Pane  | |
        # |  ==================  +-------------+ |
        # +--------------------------------------+
        self._grid_view = GridRenderPanel()
        content_row.addWidget(self._grid_view, 3)

        # ===========================
        # ||  Right: Control Pane  ||
        # || +-------------------+ ||
        # || | Episodes          | ||
        # || +-------------------+ ||
        # || | Visualization     | ||
        # || +-------------------+ ||
        # ===========================
        right_pane = QWidget()
        right_pane_layout = QVBoxLayout(right_pane)
        right_pane_layout.setSpacing(18)
        right_pane_layout.setContentsMargins(0, 0, 0, 0)

        # +-------------------------+
        # |   Right: Control Pane   |
        # |  =====================  |
        # |  || Episodes List   ||  |
        # |  =====================  |
        # |  +-------------------+  |
        # |  | Visualization     |  |
        # |  +-------------------+  |
        # +-------------------------+
        group_box = QGroupBox("Episodes")
        layout = QVBoxLayout(group_box)
        self._episode_list = QListWidget()
        self._episode_list.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self._episode_list)
        right_pane_layout.addWidget(group_box)
        
        # +-------------------------+
        # |   Right: Control Pane   |
        # |  +-------------------+  |
        # |  | Episodes          |  |
        # |  +-------------------+  |
        # |  =====================  |
        # |  || Visualization   ||  |
        # |  =====================  |
        # +-------------------------+
        self._visualization_panel = VisualizationPanel()
        right_pane_layout.addWidget(self._visualization_panel)
        right_pane_layout.addStretch(1)
        content_row.addWidget(right_pane, 2)

        # +--------------------------------------+
        # |               Root Layout            |
        # |  +--------------------------------+  |
        # |  | Content Row                    |  |
        # |  +--------------------------------+  |
        # |  ==================================  |
        # |  || Step Slider                  ||  |
        # |  ==================================  |
        # +--------------------------------------+
        self._step_slider = QSlider(Qt.Horizontal)
        self._step_slider.setFixedHeight(30)
        self._step_slider.setFocusPolicy(Qt.NoFocus)
        self._step_slider.setStyleSheet(
            """
            QSlider::groove:horizontal { background: #071427; height: 10px; border-radius: 5px; }
            QSlider::handle:horizontal { background: #6b9fd6; width: 30px; border-radius: 15px; margin: -10px 0; }
            """
        )
        root_layout.addWidget(self._step_slider)

        self.setCentralWidget(root)

    # ====================
    # Binding
    # ====================
    def _bind_presenter(self) -> None:
        self._episode_list.currentRowChanged.connect(self._on_episode_selected)
        self._step_slider.valueChanged.connect(self._on_step_slider_changed)
        self._visualization_panel.bind_trail_toggled(self._on_trail_toggled)
        self._visualization_panel.bind_q_table_toggled(self._on_q_table_toggled)
        
    def _on_episode_selected(self, episode_index: int) -> None:
        min_step, max_step, step = self._presenter.on_episode_selected(episode_index)
        self._step_slider.blockSignals(True)
        self._step_slider.setRange(min_step, max_step)
        self._step_slider.setValue(step)
        self._step_slider.blockSignals(False)
        self._sync_grid_view()

    def _on_step_slider_changed(self, step: int) -> None:
        self._presenter.on_step_slider_changed(step)
        self._sync_grid_view()
        
    def _sync_grid_view(self) -> None:
        episode_id, step_index = self._presenter.current_render_target()
        frame = self._dataset_service.render_episode_step(episode_id=episode_id, step_index=step_index)
        self._grid_view.set_frame(frame)

    def _on_trail_toggled(self, enabled: bool) -> None:
        self._dataset_service.set_trail_enabled(enabled)
        self._sync_grid_view()

    def _on_q_table_toggled(self, enabled: bool) -> None:
        self._dataset_service.set_distribution_enabled(enabled)
        self._sync_grid_view()

    # ====================
    # Qt Native Events
    # ====================
    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()

        if key == Qt.Key_Up:
            row = max(0, self._episode_list.currentRow() - 1)
            self._episode_list.setCurrentRow(row)
            event.accept()
            return

        if key == Qt.Key_Down:
            row = min(self._episode_list.count() - 1, self._episode_list.currentRow() + 1)
            self._episode_list.setCurrentRow(row)
            event.accept()
            return

        if key == Qt.Key_Left:
            self._step_slider.setValue(self._step_slider.value() - 1)
            event.accept()
            return

        if key == Qt.Key_Right:
            self._step_slider.setValue(self._step_slider.value() + 1)
            event.accept()
            return

        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._dataset_service.close()
        super().closeEvent(event)
