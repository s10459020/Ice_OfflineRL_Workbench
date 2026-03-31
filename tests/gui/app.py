from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ice_offline.env.gui.presenters import ViewerPresenter
from ice_offline.env.gui.services import MinariDatasetService
from ice_offline.env.gui.views.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    dataset_id = "test_convert_fullobs-v0"
    service = MinariDatasetService(dataset_id=dataset_id)
    episodes = service.list_episodes()
    presenter = ViewerPresenter(episodes)
    window = MainWindow(presenter=presenter, dataset_service=service)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
