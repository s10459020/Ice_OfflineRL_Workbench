import sys

from PySide6.QtWidgets import QApplication

from ice_offline.gui.models.model_actor import ActorModel
from ice_offline.gui.viewmodels.viewmodel_actor import ActorViewModel
from ice_offline.gui.views.window_actor import MainWindow


def main() -> None:
    app = QApplication(sys.argv)

    model = ActorModel()
    viewmodel = ActorViewModel(model)
    window = MainWindow(viewmodel)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
