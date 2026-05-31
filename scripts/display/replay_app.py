import sys

from PySide6.QtWidgets import QApplication

from ice_offline.gui.models.model_replay import ReplayModel
from ice_offline.gui.viewmodels.viewmodel_replay import ReplayViewModel
from ice_offline.gui.views.window_replay import MainWindow
from ice_offline.data.state.hopper import HopperConverter, HopperState, HopperStateIO


def main() -> None:
    app = QApplication(sys.argv)

    model = ReplayModel(
        state_cls=HopperState, 
        state_io_cls=HopperStateIO,
        state_converter_cls=HopperConverter, 
    )
    viewmodel = ReplayViewModel(model)
    window = MainWindow(viewmodel)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()