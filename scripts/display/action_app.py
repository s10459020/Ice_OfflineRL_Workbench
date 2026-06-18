import sys

from PySide6.QtWidgets import QApplication

from ice_offline.gui.models.model_action import ActionModel
from ice_offline.gui.viewmodels.viewmodel_action import ActionViewModel
from ice_offline.gui.views.window_action import MainWindow


def main() -> None:
    app = QApplication(sys.argv)

    model = ActionModel()
    viewmodel = ActionViewModel(model)
    window = MainWindow(viewmodel)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
