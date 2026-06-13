import os
from dataclasses import dataclass

import minari

from ice_offline.config.paths import DATASETS_ROOT
from ice_offline.dataset.base import Dataset


@dataclass
class MinariDataset(Dataset):
    minari_dataset_id: str = ""

    def __post_init__(self) -> None:
        self.ensure_dataset()
        super().__post_init__()

    def ensure_dataset(self) -> None:
        if self.path.exists():
            return
        if not self.minari_dataset_id:
            return
        os.environ.setdefault("MINARI_DATASETS_PATH", str(DATASETS_ROOT))
        minari.load_dataset(self.minari_dataset_id, download=True)
