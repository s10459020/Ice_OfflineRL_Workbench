"""Entrypoint for the dataset GUI prototype."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

# ====================
# Main Window Import
# ====================
if __package__:
    from .presenters import ViewerPresenter
    from .services import MinariDatasetService
    from .views.main_window import MainWindow
else:
    # Allow direct execution: python source/ice_offline/gui/app.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ice_offline.gui.presenters import ViewerPresenter
    from ice_offline.gui.services import MinariDatasetService
    from ice_offline.gui.views.main_window import MainWindow


def main() -> int:
    # Create and run a single Qt application event loop.
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
