from dataclasses import dataclass
from pathlib import Path

from ice_offline.dataset._spec import Dataset
from ice_offline.store.d4rl.loader import D4rlLoader
from ice_offline.config.paths import DATASETS_ROOT


@dataclass
class HopperMediumD4rlDataset(Dataset):
    id: str = "hopper_medium_d4rl"
    env_id: str = "Hopper-v5"
    path: Path = DATASETS_ROOT / Path("d4rl/hopper_medium-v2/d4rl_data.hdf5")

    def make_loader(self):
        return D4rlLoader(self.path, device=self.device)


