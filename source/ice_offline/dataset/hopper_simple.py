from dataclasses import dataclass
from pathlib import Path

from ice_offline.dataset._spec import Dataset
from ice_offline.tools.paths import minari_root


@dataclass
class HopperSimpleDataset(Dataset):
    id: str = "hopper_simple"
    env_id: str = "Hopper-v5"
    dataset_path: str | Path = minari_root() / Path("mujoco/hopper/simple-v0/data/main_data.hdf5")

    def load(self, dataset_path: str | Path = dataset_path):
        return super().load(dataset_path)
