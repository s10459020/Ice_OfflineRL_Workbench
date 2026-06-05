from dataclasses import dataclass
from pathlib import Path
import os

import minari

from ice_offline.dataset._spec import Dataset
from ice_offline.tools.paths import minari_root


@dataclass
class Walker2dSimpleDataset(Dataset):
    id: str = "walker2d_simple"
    env_id: str = "Walker2d-v5"
    minari_dataset_id: str = "mujoco/walker2d/simple-v0"
    dataset_path: str | Path = minari_root() / Path("mujoco/walker2d/simple-v0/data/main_data.hdf5")

    def load(self, dataset_path: str | Path | None = None):
        if dataset_path is None:
            self.ensure_dataset()
            dataset_path = self.dataset_path
        return super().load(dataset_path)

    def ensure_dataset(self) -> None:
        if Path(self.dataset_path).exists():
            return
        os.environ.setdefault("MINARI_DATASETS_PATH", str(minari_root()))
        minari.load_dataset(self.minari_dataset_id, download=True)
