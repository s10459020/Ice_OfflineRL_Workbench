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

from ice_offline.config.paths import RUNS_ROOT
from ice_offline.gui.views.widget_plot import PlotWidget
from ice_offline.gui.views.widget_select import SelectWidget
from ice_offline.gui.views.widget_slider import SliderWidget


class MainWindow(QMainWindow):
    def __init__(self, viewmodel) -> None:
        super().__init__()
        self._viewmodel = viewmodel

        self._run_button_1 = QPushButton("Select Prob 1")
        self._run_button_2 = QPushButton("Select Prob 2")
        self._select = SelectWidget()
        self._slider = SliderWidget()
        self._plot = PlotWidget()

        self.setWindowTitle("Actor Viewer")
        self.resize(1080, 720)
        self.setFocusPolicy(Qt.StrongFocus)

        self._build_ui()
        self._bind_events()
        self._apply_loaded_state(self._viewmodel.initial_state())

    def _apply_loaded_state(self, state) -> None:
        self._select.set_title(state.select_title)
        self._select.set_labels(state.select_labels)
        self._select.set_index(state.select_index)
        self._slider.set_range(0, state.slider_max, state.slider_value)
        self._plot.set_axis_names(state.x_name, state.y_name)
        self._plot.set_curves(state.curves_1, state.curves_2)

    def _apply_episode_state(self, state) -> None:
        self._select.set_index(state.select_index)
        self._slider.set_range(0, state.slider_max, state.slider_value)
        self._plot.set_curves(state.curves_1, state.curves_2)

    def _apply_frame_state(self, state) -> None:
        self._slider.set_value(state.slider_value)
        self._plot.set_curves(state.curves_1, state.curves_2)

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

        content_layout.addWidget(self._plot, 3)

        control_layout = QVBoxLayout()
        control_layout.setSpacing(18)
        control_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addLayout(control_layout, 1)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        button_layout.addWidget(self._run_button_1)
        button_layout.addWidget(self._run_button_2)
        control_layout.addLayout(button_layout)
        control_layout.addWidget(self._select)
        control_layout.addStretch(1)

    def _bind_events(self) -> None:
        self._run_button_1.clicked.connect(self._on_run_selected_1)
        self._run_button_2.clicked.connect(self._on_run_selected_2)
        self._select.selected.connect(self._on_selected)
        self._slider.changed.connect(self._on_slided)

    def _on_run_selected_1(self):
        initial_dir = str(RUNS_ROOT.resolve())
        path = QFileDialog.getExistingDirectory(self, "Select Probe 1", initial_dir)
        if not path:
            return
        try:
            self._apply_loaded_state(self._viewmodel.load_run_data(path, 1))
        except Exception:
            print("load failed:")
            traceback.print_exc()

    def _on_run_selected_2(self):
        initial_dir = str(RUNS_ROOT.resolve())
        path = QFileDialog.getExistingDirectory(self, "Select Probe 2", initial_dir)
        if not path:
            return
        try:
            self._apply_loaded_state(self._viewmodel.load_run_data(path, 2))
        except Exception:
            print("load failed:")
            traceback.print_exc()

    def _on_selected(self, index: int) -> None:
        self._apply_episode_state(self._viewmodel.set_episode(index))

    def _on_slided(self, value: int) -> None:
        self._apply_frame_state(self._viewmodel.set_step(value))

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key_Up:
            new_index = max(0, self._select.index() - 1)
            self._apply_episode_state(self._viewmodel.set_episode(new_index))
            event.accept()
            return
        if key == Qt.Key_Down:
            new_index = self._select.index() + 1
            self._apply_episode_state(self._viewmodel.set_episode(new_index))
            event.accept()
            return
        if key == Qt.Key_Left:
            self._apply_frame_state(self._viewmodel.set_step(self._slider.value() - 1))
            event.accept()
            return
        if key == Qt.Key_Right:
            self._apply_frame_state(self._viewmodel.set_step(self._slider.value() + 1))
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._viewmodel.close()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    from ice_offline.gui.models.model_actor import ActorModel
    from ice_offline.gui.viewmodels.viewmodel_actor import ActorViewModel

    app = QApplication(sys.argv)
    model = ActorModel()
    viewmodel = ActorViewModel(model)
    window = MainWindow(viewmodel)
    window.show()
    sys.exit(app.exec())
