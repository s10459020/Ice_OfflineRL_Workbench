
import sys

from PySide6.QtWidgets import QApplication

from ice_offline.env.gui.presenters.viewer_presenter import ViewerPresenter
from ice_offline.env.gui.services.minari_dataset_service import MinariDatasetService
from ice_offline.env.gui.views.main_window import MainWindow


def main() -> int:
    dataset_id = "retrain_agent_collect-v0"
    distribution_style = "ring"  # "ring" or "rect"
    #dataset_id = "train_agent_q_table_state_value-v0"
    #dataset_id = "test_collect_value-v0"

    app = QApplication(sys.argv)
    service = MinariDatasetService(dataset_id=dataset_id, distribution_style=distribution_style)
    episodes = service.list_episodes()
    presenter = ViewerPresenter(episodes)
    window = MainWindow(presenter=presenter, dataset_service=service)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

