from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ice_offline.env.gui.presenters import ViewerPresenter
from ice_offline.env.gui.services import MinariDatasetService
from ice_offline.env.gui.views.main_window import MainWindow


def main() -> int:
    dataset_id = "train_Step_collect-v0"
    distribution_style = "rect"  # "ring" or "rect"

    app = QApplication(sys.argv)
    service = MinariDatasetService(dataset_id=dataset_id, distribution_style=distribution_style)
    episodes = service.list_episodes()
    presenter = ViewerPresenter(episodes)
    window = MainWindow(presenter=presenter, dataset_service=service)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
