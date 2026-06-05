from pathlib import Path
from typing import Protocol

from ice_offline.dataset._types import Buffer, Episode, Metadata


class DatasetLoader(Protocol):
    # ====================
    # Dataset identity
    # ====================
    path: Path
    device: str

    # ====================
    # Loading
    # ====================
    def load_buffer(self) -> Buffer:
        ...

    def load_episodes(self) -> list[Episode]:
        ...

    def load_metadata(self) -> Metadata:
        ...
