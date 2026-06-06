from dataclasses import dataclass
from pathlib import Path
import os

import minari

from ice_offline.dataset._spec import Dataset
from ice_offline.config.paths import DATASETS_ROOT


@dataclass
class HopperMediumDataset(Dataset):
    id: str = "hopper_medium"
    env_id: str = "Hopper-v5"
    minari_dataset_id: str = "mujoco/hopper/medium-v0"
    path: Path = DATASETS_ROOT / Path("mujoco/hopper/medium-v0/data/main_data.hdf5")

    def __post_init__(self) -> None:
        self.ensure_dataset()
        super().__post_init__()

    def ensure_dataset(self) -> None:
        if self.path.exists():
            return
        os.environ.setdefault("MINARI_DATASETS_PATH", str(DATASETS_ROOT))
        minari.load_dataset(self.minari_dataset_id, download=True)



