from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ice_offline.gui.viewmodels.replay_mojoco import ViewerPresenter
from ice_offline.gui.views.widget_render import RenderWidget
from ice_offline.gui.views.widget_select import SelectWidget
from ice_offline.gui.views.widget_slider import SliderWidget
from ice_offline.tools.paths import minari_root


class MainWindow(QMainWindow):
    # ====================
    # init & reset
    # ====================
    def __init__(self, viewmodel) -> None:
        super().__init__()
        self._viewmodel = viewmodel

        self._button = QPushButton()
        self._select = SelectWidget()
        self._slider = SliderWidget()
        self._render = RenderWidget()
        self._setting =  QWidget()

        self.setWindowTitle("Replay Viewer")
        self.resize(980, 640)
        self.setFocusPolicy(Qt.StrongFocus)

        self._build_ui()
        self._bind_events()
        self._apply_state(self._viewmodel.initial_state())

    def _apply_state(self, state) -> None:
        self._button.setText(state.button_label)
        self._select.set_title(state.select_title)
        self._select.set_labels(state.episode_labels)
        self._select.set_index(state.selected_episode)

        self._slider.set_range(0, state.max_step, state.selected_step)

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

        root_layout.addWidget(self._slider)

        # +------------------------------------------+
        # |              Content Layout              |
        # |  +---------------+   +-----------------+ |
        # |  | Grid Render   |   | Control Layout  | |
        # |  +---------------+   +-----------------+ |
        # +------------------------------------------+
        content_layout.addWidget(self._render, 3)

        control_layout = QVBoxLayout()  
        control_layout.setSpacing(18)
        control_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addLayout(control_layout, 2)

        # +--------------------------+
        # |   Right: Control Layout  |
        # |  +-------------------+   |
        # |  | Push Button       |   |
        # |  +-------------------+   |
        # |  +-------------------+   |
        # |  | Select List       |   |
        # |  +-------------------+   |
        # |  +-------------------+   |
        # |  | Plugin Area       |   |
        # |  +-------------------+   |
        # +--------------------------+
        control_layout.addWidget(self._button)
        control_layout.addWidget(self._select)
        control_layout.addWidget(self._setting)
        control_layout.addStretch(1)

    # ====================
    # Widget Events
    # ====================
    def _bind_events(self) -> None:
        self._button.clicked.connect(self._on_button)
        self._select.selected.connect(self._on_selected)
        self._slider.changed.connect(self._on_slided)

    def _on_button(self):
        initial_dir = str(minari_root().resolve())
        path, _ = QFileDialog.getOpenFileName(self, "Select Dataset", initial_dir, "All Files (*.*)")
        if not path:
            return
        
        self._apply_state(self._viewmodel.load_dataset(path))

    def _on_selected(self, index: int) -> None:
        self._apply_state(self._viewmodel.set_episode(index))

    def _on_slided(self, value: int) -> None:
        self._apply_state(self._viewmodel.set_step(value))

    # ====================
    # Qt Native Events
    # ====================
    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()

        if key == Qt.Key_Up:
            self._select.set_index(max(0, self._select.index() - 1))
            event.accept()
            return

        if key == Qt.Key_Down:
            self._select.set_index(self._select.index() + 1)
            event.accept()
            return

        if key == Qt.Key_Left:
            self._slider.set_value(self._slider.value() - 1)
            event.accept()
            return

        if key == Qt.Key_Right:
            self._slider.set_value(self._slider.value() + 1)
            event.accept()
            return

        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._presenter.close()
        super().closeEvent(event)
