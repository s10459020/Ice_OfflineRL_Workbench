"""Main window for a frontend-only dataset viewer MVP mockup."""


from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ice_offline.gui.presenters.viewer_presenter import ViewerPresenter
from ice_offline.gui.services.minari_dataset_service import MinariDatasetService
from ice_offline.gui.views.panel_render import GridRenderPanel
from ice_offline.gui.views.panel_setting_minigrid import VisualizationPanel
from ice_offline.tools.paths import minari_root


class MainWindow(QMainWindow):
    # ====================
    # init & reset
    # ====================
    def __init__(self, presenter: ViewerPresenter) -> None:
        super().__init__()
        self._presenter = presenter
        self._list: QListWidget | None = None
        self._slider: QSlider | None = None
        self._render: GridRenderPanel | None = None
        self._setting: VisualizationPanel | None = None
        self._button: QPushButton | None = None

        self.setWindowTitle("Dataset GUI Mockup")
        self.resize(980, 640)
        self.setFocusPolicy(Qt.StrongFocus)
        self._build_ui()
        self._bind_events()
        self.reset_all()

    def reset_all(self) -> None:
        self._on_list_selected(0)
        self._on_slider_changed(0)
        self._setting.reset()



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
        self._render = GridRenderPanel()
        content_layout.addWidget(self._render, 3)

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
        self._button = QPushButton("Select Dataset")
        control_layout.addWidget(self._button)

        self._list = QListWidget()
        self._list.setFocusPolicy(Qt.NoFocus)
        group_box = QGroupBox("Episodes")
        layout = QVBoxLayout(group_box)
        layout.addWidget(self._list)
        control_layout.addWidget(group_box)
        
        self._setting = VisualizationPanel()
        control_layout.addWidget(self._setting)
        control_layout.addStretch(1)

    # ====================
    # Widget Events
    # ====================
    def _bind_events(self) -> None:
        self._button.clicked.connect(self._on_button_click)
        self._list.currentRowChanged.connect(self._on_list_selected)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._setting.on_changed(self._presenter._on_setting_changed)

    def _on_button_click(self):
        initial_dir = str(minari_root().resolve())
        path, _ = QFileDialog.getOpenFileName(self, "Select Dataset", initial_dir, "All Files (*.*)")
        if not path:
            return
        
        indices = self._presenter.load_dataset(path)
        self._list.blockSignals(True)
        self._list.clear()
        for index in indices:
            self._list.addItem(QListWidgetItem(str(index)))

        self._list.blockSignals(False)
        self._list.setCurrentRow(0)
        self._on_list_selected(0)

    def _on_list_selected(self, index: int) -> None:
        steps = self._presenter.set_episode(index)
        self._slider.blockSignals(True)
        self._slider.setRange(0, steps)
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._sync_render()

    def _sync_render(self) -> None:
        frame = self._presenter.render()
        self._render.set_frame(frame)

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
