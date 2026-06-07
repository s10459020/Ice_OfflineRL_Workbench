from dataclasses import dataclass
from pathlib import Path

from ice_offline.config.paths import custom_dataset_path
from ice_offline.dataset._spec import Dataset


@dataclass
class HopperSimpleOneDataset(Dataset):
    id: str = "hopper_simple_one"
    env_id: str = "Hopper-v5"
    path: Path = custom_dataset_path(id)
