from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGroupBox, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

class SelectWidget(QWidget):
    selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._list = QListWidget()
        self._list.setFocusPolicy(Qt.NoFocus)

        self._group = QGroupBox("Title")
        group_layout = QVBoxLayout(self._group)
        group_layout.addWidget(self._list)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._group)

        self._list.currentRowChanged.connect(self.selected.emit)


    # ====================
    # Public API
    # ====================
    def set_title(self, title: str) -> None:
        self._group.setTitle(title)

    def set_labels(self, labels: list[str]) -> None:
        self._list.blockSignals(True)   
        self._list.clear()

        for label in labels:
            self._list.addItem(QListWidgetItem(label))

        self._list.blockSignals(False)

    def set_index(self, index: int) -> None:
        self._list.blockSignals(True)   
        self._list.setCurrentRow(index)
        self._list.blockSignals(False)

    def index(self) -> int:
        return self._list.currentRow()


