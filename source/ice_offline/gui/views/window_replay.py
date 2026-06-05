import traceback

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

from ice_offline.gui.views.widget_render import RenderWidget
from ice_offline.gui.views.widget_select import SelectWidget
from ice_offline.gui.views.widget_setting import SettingWidget
from ice_offline.gui.views.widget_slider import SliderWidget
from ice_offline.tools.paths import dataset_root


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
        self._setting = SettingWidget()

        self.setWindowTitle("Replay Viewer")
        self.resize(980, 640)
        self.setFocusPolicy(Qt.StrongFocus)

        self._build_ui()
        self._bind_events()
        self._apply_state(self._viewmodel.initial_state())

    def _apply_state(self, state) -> None:
        self._button.setText(state.button_label)
        self._select.set_title(state.select_title)
        self._select.set_labels(state.select_labels)
        self._select.set_index(state.select_index)
        self._slider.set_range(0, state.slider_max, state.slider_value)
        self._setting.set_value(state.step_jump)
        if state.frame is not None:
            self._render.set_frame(state.frame)

    # ====================
    # UI
    # ====================
    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(20)
        self.setCentralWidget(root)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(24)
        root_layout.addLayout(content_layout, 1)

        root_layout.addWidget(self._slider)
        content_layout.addWidget(self._render, 3)

        control_layout = QVBoxLayout()
        control_layout.setSpacing(18)
        control_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addLayout(control_layout, 2)

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
        self._setting.changed.connect(self._on_setting_changed)

    def _on_button(self):
        initial_dir = str(dataset_root().resolve())
        path, _ = QFileDialog.getOpenFileName(self, "Select Dataset", initial_dir, "All Files (*.*)")
        if not path:
            return

        print(f"[window] load button path={path}")
        try:
            state = self._viewmodel.load_dataset(path)
            print("[window] viewmodel.load_dataset done")
            self._apply_state(state)
            print("[window] apply_state done")
        except Exception:
            print("[window] load failed:")
            traceback.print_exc()

    def _on_selected(self, index: int) -> None:
        self._apply_state(self._viewmodel.set_episode(index))

    def _on_slided(self, value: int) -> None:
        self._apply_state(self._viewmodel.set_step(value))

    def _on_setting_changed(self, value: int) -> None:
        self._apply_state(self._viewmodel.set_step_jump(value))

    # ====================
    # Qt Native Events
    # ====================
    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()

        if key == Qt.Key_Up:
            new_index = max(0, self._select.index() - 1)
            self._apply_state(self._viewmodel.set_episode(new_index))
            event.accept()
            return

        if key == Qt.Key_Down:
            new_index = self._select.index() + 1
            self._apply_state(self._viewmodel.set_episode(new_index))
            event.accept()
            return

        step_jump = self._viewmodel.step_jump()

        if key == Qt.Key_Left:
            new_step = self._slider.value() - step_jump
            self._apply_state(self._viewmodel.set_step(new_step))
            event.accept()
            return

        if key == Qt.Key_Right:
            new_step = self._slider.value() + step_jump
            self._apply_state(self._viewmodel.set_step(new_step))
            event.accept()
            return

        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._viewmodel.close()
        super().closeEvent(event)
