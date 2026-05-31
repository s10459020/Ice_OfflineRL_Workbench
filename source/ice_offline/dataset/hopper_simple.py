from dataclasses import dataclass
import os
from pathlib import Path

import minari

from ice_offline.dataset._spec import Dataset
from ice_offline.tools.paths import minari_root


@dataclass
class HopperSimpleDataset(Dataset):
    id: str = "hopper_simple"
    env_id: str = "Hopper-v5"
    minari_dataset_id: str = "mujoco/hopper/simple-v0"
    dataset_path: str | Path = minari_root() / Path("mujoco/hopper/simple-v0/data/main_data.hdf5")

    def load(self):
        self.ensure_dataset()
        return super().load(self.dataset_path)

    def ensure_dataset(self) -> None:
        if Path(self.dataset_path).exists():
            return
        os.environ.setdefault("MINARI_DATASETS_PATH", str(minari_root()))
        minari.load_dataset(self.minari_dataset_id, download=True)
