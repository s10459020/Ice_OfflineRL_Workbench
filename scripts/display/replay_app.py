import sys

from PySide6.QtWidgets import QApplication

from ice_offline.gui.models.model_replay import ReplayModel
from ice_offline.gui.viewmodels.viewmodel_replay import ReplayViewModel
from ice_offline.gui.views.window_replay import MainWindow


def main() -> None:
    app = QApplication(sys.argv)

    model = ReplayModel()
    viewmodel = ReplayViewModel(model)
    window = MainWindow(viewmodel)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
